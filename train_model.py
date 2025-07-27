from MLA import MLA
from MTP import MTP
from MOE import MOE
from MLA import Config
import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
import math
from transformers import GPT2Tokenizer
import gc 
import os

# Add at the beginning of your script
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cudnn.benchmark = False

config = Config()
tokenizer = GPT2Tokenizer.from_pretrained(config.model_name)
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

print(f"using device={device}")

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.layerNormMla = nn.LayerNorm(config.n_embd)
        self.mla = MLA(config, device)
        self.layerNormMoe = nn.LayerNorm(config.n_embd)
        self.moe = MOE()
    def forward(self, x, kr_cache=None, starting_pos=0):
        # Pass kr_cache through the attention layer
        mla_out, new_kr_cache = self.mla(self.layerNormMla(x), kr_cache, starting_pos)
        x = x + mla_out
        x = x + self.moe(self.layerNormMoe(x))
        return x, new_kr_cache


# from datasets import load_dataset
# ds = load_dataset("wikipedia", "20220301.en", split="train[:2%]")

# print("Number of samples in 2% Wikipedia:", len(ds))

# # Join all article texts with a separator
# full_text = "\n\n".join(ds[i]["text"] for i in range(len(ds)))

# # Optional: Save to file
# with open("wikipedia_2percent.txt", "w", encoding="utf-8") as f:
#     f.write(full_text)

# with open("wikipedia_2percent.txt", "r", encoding="utf-8") as f:
#     full_text = f.read()

# print(f"total len of training data = {len(full_text)}")
# full_text = full_text[:10000]
# print(len(full_text))

# print(f"Length of text: {len(full_text)} characters")
# print(full_text[:500])  # print first 500 characters

# data = tokenizer.encode(full_text)
# Save as binary tensor (recommended for large datasets)
# torch.save(torch.tensor(data, dtype=torch.long), "tokenized_data.pt")
  
# data = torch.load("tokenized_data.pt")  # already a tensoryes 
data = torch.load('child_stories.pt')


print(f"len of data is {len(data)}")


class dataLoaderLite:
    def __init__(self):
        self.current_pos = 0
    def reset(self):
        self.current_pos = 0
    def next_batch(self):
        dt = data[self.current_pos: self.current_pos + config.batch_size * config.block_size]
        dt = dt.detach().clone().long()
        self.current_pos += config.batch_size * config.block_size
        if self.current_pos + config.batch_size * config.block_size > len(data):
            self.reset()
        return dt.view(config.batch_size, config.block_size)
    
dataLoader = dataLoaderLite()

class Deepseek(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([TransformerBlock() for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = MTP()
        # Share embedding weights with MTP
        self.lm_head.token_embedding = self.transformer.wte
        self.lm_head.single_token_head.weight.data = self.transformer.wte.weight.data.clone()
        # weight initialization 
        self.apply(self.weight_init_)

    def weight_init_(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            if hasattr(module, "NANOGPT_SCALE"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
    
    def forward(self, x, targets=None, inference_mode=False, kr_caches=None, starting_pos=0):
        B, T = x.shape
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        tok_embd = self.transformer.wte(x) # (B, T, C)
        hidden = tok_embd
        # Initialize kr_caches if None
        if kr_caches is None:
            kr_caches = [None] * len(self.transformer.h)
        new_kr_caches = []
        
        for i, h in enumerate(self.transformer.h):
            hidden, new_kr_cache = h(hidden, kr_caches[i], starting_pos)
            new_kr_caches.append(new_kr_cache)
            
        hidden = self.transformer.ln_f(hidden)
        # Use inference_mode flag to determine which head to use
        logits = self.lm_head(hidden, input_tokens=x, device=device, inference_mode=inference_mode)
        loss = None 
        if targets is not None:
            # logits shape: [batch, seq_len-no_head_in_mtp, no_head_in_mtp, vocab_size]
            # targets shape: [batch, seq_len-no_head_in_mtp, no_head_in_mtp]
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = targets.view(-1)
            # print(logits_flat.shape)
            # print(targets_flat.shape)
            loss = F.cross_entropy(logits_flat, targets_flat)
        
        return logits, loss, new_kr_caches
    
    def configure_optimizer(self, lr, decay, device):
        param_group = {pn:p for pn, p in self.named_parameters()}
        param_group = {pn: p for pn, p in param_group.items() if p.requires_grad}

        decay_params = [p for pn, p in param_group.items() if p.dim() >= 2]
        non_decay_params = [p for pn, p in param_group.items() if p.dim() < 2]

        optim_group = [
            {"params": decay_params, "weight_decay": decay}, 
            {"params": non_decay_params, "weight_decay": 0}
        ]
        # Use 8-bit optimizer for memory efficiency if available
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(optim_group, betas=(0.9, 0.95), lr=lr, eps=1e-8)
            print("Using 8-bit optimizer for memory efficiency")
        except ImportError:
            fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
            is_fused = "cuda" in device and fused_available
            optimizer = torch.optim.AdamW(optim_group, betas=(0.9, 0.95), lr=lr, eps=1e-8, fused=is_fused)
            print("Using standard AdamW optimizer")
        return optimizer


max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 500
max_steps = config.max_steps
total_grad_steps = 1 << 11
weight_decay = 0.1
grad_accum_steps = total_grad_steps // (config.block_size * config.batch_size)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

torch.set_float32_matmul_precision('high')

torch.cuda.empty_cache()
gc.collect()

raw_model = Deepseek()
optimizer = raw_model.configure_optimizer(max_lr, 0.1, device)
model = raw_model
# model = torch.compile(model)
model.to(device)



def get_mtp_targets(X):
    """
    Generate targets for Multi-Token Prediction
    X: [batch, seq_len]
    Returns: [batch, seq_len-no_head_in_mtp, no_head_in_mtp]
    """
    B, T = X.shape
    targets = []
    
    for i in range(T - config.no_head_in_mtp):
        # For position i, predict next no_head_in_mtp tokens
        future_tokens = X[:, i+1:i+1+config.no_head_in_mtp]
        targets.append(future_tokens)
    
    return torch.stack(targets, dim=1)  # [batch, seq_len-no_head_in_mtp, no_head_in_mtp]

def train_model(): 
    print("model training starts")
    print(f"total grad accum steps={grad_accum_steps}")
    dataLoader.reset()
    model.train()
    total_steps_to_run = config.max_steps
    optimizer.zero_grad()
    for i in range(total_steps_to_run):
        avg_loss = 0.0
        
        for j in range(grad_accum_steps):
            X = dataLoader.next_batch()
            X = X.to(device)
            targets = get_mtp_targets(X)
            targets = targets.to(device)
            # att_masks = (X != tokenizer.eos_token_id).long()
            # att_masks = att_masks.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss, _ = model(X, targets=targets)
            # print(f"shape of logits = {logits.shape}")
            if not torch.isfinite(loss):
                print(f"Non-finite loss detected at step {i}: {loss}")
                break  # Or skip step / reload checkpoint


            loss = loss / grad_accum_steps
            avg_loss += loss.detach()
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        for param_group in optimizer.param_groups:
            param_group['lr'] = get_lr(i)
        optimizer.step()
        print(f"loss at step {i} is {avg_loss}")
        # Clear cache
        del X, targets, logits, loss
        if device == 'cuda' and i % 20 == 0:
            torch.cuda.empty_cache()
            gc.collect()
        optimizer.zero_grad()


train_model()

torch.save({
    'model_state_dict': raw_model.state_dict(), 
    'config': config
}, 'trained_model_deepseek.pth')
