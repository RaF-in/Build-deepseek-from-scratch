import torch
from dataclasses import dataclass
import torch.nn as nn
from ROPE import RotaryPositionalEmbeddings
import torch.nn.functional as F
import math

@dataclass
class Config:
    block_size: int = 1024
    batch_size: int = 1
    n_embd: int = 768
    n_head: int = 4
    head_dim: int = n_embd // n_head
    q_lora_rank: int = n_embd // 2
    kv_lora_rank: int = (2 * n_embd) // 3
    qk_nope_rank: int = head_dim // 2
    qk_rope_rank: int = head_dim // 2
    v_head_dim: int = head_dim // 2
    num_of_experts: int = 8
    num_of_shared_experts: int = 2
    activated_experts: int = 4
    no_head_in_mtp: int = 2
    vocab_size: int = 50257
    n_layer: int = 4
    model_name = 'gpt2'
    max_steps: int = 500


class MLA(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        # Q projections
        self.Q_down_proj = nn.Linear(config.n_embd, config.q_lora_rank, bias=False)
        self.Q_up_proj = nn.Linear(config.q_lora_rank, config.n_head * (config.qk_nope_rank + config.qk_rope_rank))
        self.Q_layer_norm = nn.LayerNorm(config.q_lora_rank)
        #K_V projections
        self.KV_down_proj = nn.Linear(config.n_embd, config.kv_lora_rank + config.qk_rope_rank, bias=False)
        self.KV_up_proj = nn.Linear(config.kv_lora_rank, config.n_head * (config.qk_nope_rank + config.v_head_dim))
        self.KV_layer_norm = nn.LayerNorm(config.kv_lora_rank)

        self.Wo = nn.Linear(config.n_head * config.v_head_dim, config.n_embd, bias=False)

        # Rope precompute
        self.rope = RotaryPositionalEmbeddings(config.block_size, config.qk_rope_rank, 10000)
        self.freqs = self.rope.precompute_complex_rope(device)

        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size).view(1, 1, config.block_size, config.block_size)))
        self.Wo.NANOGPT_SCALE = 1

    def forward(self, x, kr_cache=None, starting_pos=0):
        # Q calculations
        B, T, C = x.shape
        compressed = self.Q_down_proj(x)
        compressed = self.Q_layer_norm(compressed)
        compressed = self.Q_up_proj(compressed) # B, T, n_head * (q_nope_rank + q_rope_rank)
        compressed = compressed.view(B, T, -1, self.config.qk_nope_rank + self.config.qk_rope_rank).transpose(1, 2) # B, n_head, T, nope + rope
        q_nope, q_rope = torch.split(compressed, [self.config.qk_nope_rank, self.config.qk_rope_rank], dim=-1)
        q_rope = self.rope.apply_rope(q_rope, self.freqs[starting_pos:starting_pos + T]) # B, n_head, T, rope
        # kv calculations
        
        kv_compressed = self.KV_down_proj(x) # B, T, kv_lora_rank + qk_rope_rank

        c_kv, k_rope = torch.split(kv_compressed, [self.config.kv_lora_rank, self.config.qk_rope_rank], dim=-1)
        c_kv = self.KV_layer_norm(c_kv) # B, T, kv_lora_rank 
        k_rope = self.rope.apply_rope(k_rope, self.freqs[starting_pos:starting_pos + T]) # B, T, qk_rope_rank
        
        # c_kv and k_rope from cache if cache is not none
        if kr_cache is not None: 
            kv_cache, k_cache = torch.split(kr_cache, [self.config.kv_lora_rank, self.config.qk_rope_rank], dim=-1)
            c_kv = torch.cat((kv_cache, c_kv), 1) # B, seq_len, kv_lora_rank
            k_rope = torch.cat((k_cache, k_rope), dim=1) # B, seq_len, qk_rope_rank
            
        new_kr_cache = torch.cat((c_kv, k_rope), dim=-1) # B, seq_len, kv_lora_rank + qk_rope_rank
        
        # kv up projections
        seq_len = c_kv.shape[1]
        kv_up = self.KV_up_proj(c_kv) # B, seq_len, n_head * (qk_nope + v_head)
        kv_up = kv_up.view(B, seq_len, self.config.n_head, self.config.qk_nope_rank+self.config.v_head_dim).transpose(1, 2) # B, n_head, seq_len, (qk_nope, v_head)
        k, v = torch.split(kv_up, [self.config.qk_nope_rank, self.config.v_head_dim], -1) # B, n_head, seq_len, qk_nope and B, n_head, seq_len, v_head
        
        # k_rope = k_rope.unsqueeze(2)
        # k_rope = k_rope.repeat(1, 1, self.config.n_head, 1).transpose(1, 2)
        # OR simply
        k_rope = k_rope.unsqueeze(1).expand(-1, self.config.n_head, -1, -1) # B, n_head, seq_len, qk_rope_rank

        q = torch.cat((q_nope, q_rope), -1) # B, n_head, T, nope + B, n_head, T, rope = B, n_head, T, nope + rope
        k = torch.cat((k, k_rope), -1) # B, n_head, seq_len, nope + rope

        attn_scores = q @ k.transpose(-2, -1) # B, n_head, T, nope + rope @ B, n_head, nope + rope, T = B, n_head, T, seq_len
        
        attn_scores.masked_fill(self.bias[:, :, starting_pos:starting_pos+T, :seq_len] == 0, float("-inf"))

        attn_scores = attn_scores / math.sqrt(k.shape[-1])

        out = F.softmax(attn_scores, -1)

        out = out @ v # B, n_head, T, seq_len @ B, n_head, seq_len, v_head = B, n_head, T, v_head

        out = out.transpose(1, 2).contiguous().view(B, T, self.config.n_head * self.config.v_head_dim) # B, T, n_head * v_head

        return self.Wo(out), new_kr_cache


