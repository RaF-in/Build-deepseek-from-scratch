import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from MLA import Config

config = Config()

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape
        
        # Create Q, K, V
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        return self.w_o(attn_output)

class TransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x
    

class MTP(nn.Module):
    def __init__(self):
        super().__init__()
        self.projections = nn.ModuleList([nn.Linear(config.n_embd * 2, config.n_embd) for _ in range(config.no_head_in_mtp)])
        self.transformer = TransformerLayer(config.n_embd, config.n_head, config.n_embd)
        self.unembd = nn.Linear(config.n_embd, config.vocab_size)
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        # Add a simple linear layer for single token prediction during inference
        self.single_token_head = nn.Linear(config.n_embd, config.vocab_size)

    def rmsnorm(self, x):
        rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True)) + 1e-8
        return x / rms

    def forward(self, x, input_tokens=None, device='cpu', inference_mode=False):

        # For inference, use simple single token prediction
        if inference_mode:
            # Use the simple linear head for single token prediction
            return self.single_token_head(x)  # [batch, seq_len, vocab_size]
        
        batch_size = x.size(0)
        

        hidden = self.transformer(x)
        
        mtp_upper = x.shape[1] - config.no_head_in_mtp
        
        # Pre-allocate output tensor to avoid dynamic memory allocation
        all_logits = torch.empty(
            batch_size, mtp_upper, config.no_head_in_mtp, config.vocab_size,
            dtype=x.dtype, device=x.device
        )
        for i in range(mtp_upper):
            h_prev = hidden[:, i, :].clone()
            curr_logits = []
            for k in range(config.no_head_in_mtp):
                future_token = i + k + 1
                if input_tokens is not None:
                    # Use the actual future token embeddings
                    future_emb = self.token_embedding(input_tokens[:, future_token])
                else:
                    # Fallback to using the input embeddings
                    future_emb = x[:, future_token, :]
                h_prev = self.rmsnorm(h_prev)
                future_emb = self.rmsnorm(future_emb)
                concatenated_res = torch.cat([future_emb, h_prev], dim=-1)
                curr_h = self.projections[k](concatenated_res)

                # Transformer forward pass with error handling
                transformer_input = curr_h.unsqueeze(1)
                curr_res = self.transformer(transformer_input)
                
                # Safe reshape with validation
                expected_size = batch_size * curr_res.size(-1)
                if curr_res.numel() != expected_size:
                    curr_res = curr_res.contiguous().view(batch_size, -1)
                else:
                    curr_res = curr_res.view(batch_size, -1)
                
                # Generate logits with gradient checkpointing to save memory
                logits = self.unembd(curr_res)
                
                # Bounds checking before storing
                if (i < all_logits.size(1) and k < all_logits.size(2) and 
                    logits.size(0) == all_logits.size(0) and 
                    logits.size(1) == all_logits.size(3)):
                    all_logits[:, i, k, :] = logits
                else:
                    raise IndexError(f"Tensor size mismatch at position [{i}, {k}]")
                
                # Update h_prev and clean up intermediate tensors
                h_prev = curr_h.detach()  # Detach to prevent gradient accumulation

                # Clear intermediate tensors to free memory
                del concatenated_res, curr_h, curr_res, logits
                
                # Force garbage collection every few iterations
                if (i * config.no_head_in_mtp + k) % 10 == 0:
                    torch.cuda.empty_cache()

        return all_logits
    
# Mtp = MTP()
# x = torch.randn(2, 1024,  768)
# print(Mtp(x, None, device='cuda').shape)


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math
# from MLA import Config

# config = Config()

# class MultiHeadAttention(nn.Module):
#     def __init__(self, d_model, n_heads):
#         super().__init__()
#         self.d_model = d_model
#         self.n_heads = n_heads
#         self.d_k = d_model // n_heads
        
#         self.w_q = nn.Linear(d_model, d_model)
#         self.w_k = nn.Linear(d_model, d_model)
#         self.w_v = nn.Linear(d_model, d_model)
#         self.w_o = nn.Linear(d_model, d_model)
        
#     def forward(self, x, mask=None):
#         batch_size, seq_len, d_model = x.shape
        
#         # Create Q, K, V
#         Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
#         K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
#         V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
#         # Attention
#         scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
#         if mask is not None:
#             scores = scores.masked_fill(mask == 0, -1e9)
            
#         attn_weights = F.softmax(scores, dim=-1)
#         attn_output = torch.matmul(attn_weights, V)
        
#         # Concatenate heads
#         attn_output = attn_output.transpose(1, 2).contiguous().view(
#             batch_size, seq_len, d_model
#         )
        
#         return self.w_o(attn_output)

# class TransformerLayer(nn.Module):
#     def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
#         super().__init__()
#         self.self_attn = MultiHeadAttention(d_model, n_heads)
#         self.feed_forward = nn.Sequential(
#             nn.Linear(d_model, d_ff),
#             nn.ReLU(),
#             nn.Linear(d_ff, d_model)
#         )
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)
        
#     def forward(self, x, mask=None):
#         # Self-attention with residual connection
#         attn_output = self.self_attn(x, mask)
#         x = self.norm1(x + self.dropout(attn_output))
        
#         # Feed-forward with residual connection
#         ff_output = self.feed_forward(x)
#         x = self.norm2(x + self.dropout(ff_output))
        
#         return x

# class MTP(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.projections = nn.ModuleList([nn.Linear(config.n_embd * 2, config.n_embd) for _ in range(config.no_head_in_mtp)])
#         self.transformer = TransformerLayer(config.n_embd, config.n_head, config.n_embd)
#         self.unembd = nn.Linear(config.n_embd, config.vocab_size)
#         self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
#         # Add a simple linear layer for single token prediction during inference
#         self.single_token_head = nn.Linear(config.n_embd, config.vocab_size)

#     def rmsnorm(self, x):
#         # Add epsilon to prevent division by zero
#         rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + 1e-8)
#         return x / rms

#     def forward(self, x, input_tokens=None, device='cpu', inference_mode=False):
#         # Input validation
#         if x.dim() != 3:
#             raise ValueError(f"Expected 3D input, got {x.dim()}D")
        
#         batch_size, seq_len, hidden_dim = x.shape
        
#         # For inference, use simple single token prediction
#         if inference_mode:
#             return self.single_token_head(x)  # [batch, seq_len, vocab_size]
        
#         # Validate sequence length
#         if seq_len <= config.no_head_in_mtp:
#             print(f"Warning: Sequence length {seq_len} too short for MTP, using single token prediction")
#             return self.single_token_head(x).unsqueeze(2)  # Add extra dimension for consistency
        
#         # Apply transformer to get enhanced hidden states
#         try:
#             hidden = self.transformer(x)
#         except Exception as e:
#             print(f"Error in transformer forward pass: {e}")
#             # Fallback to using input directly
#             hidden = x
        
#         # Calculate valid range for MTP
#         mtp_upper = seq_len - config.no_head_in_mtp
        
#         if mtp_upper <= 0:
#             print(f"Warning: Not enough tokens for MTP. seq_len={seq_len}, no_head_in_mtp={config.no_head_in_mtp}")
#             # Return dummy output with correct shape
#             return torch.zeros(batch_size, 1, config.no_head_in_mtp, config.vocab_size, 
#                              dtype=x.dtype, device=x.device)
        
#         # Pre-allocate output tensor
#         try:
#             all_logits = torch.zeros(
#                 batch_size, mtp_upper, config.no_head_in_mtp, config.vocab_size,
#                 dtype=x.dtype, device=x.device
#             )
#         except RuntimeError as e:
#             print(f"Failed to allocate output tensor: {e}")
#             # Use smaller chunks or fallback
#             return self.single_token_head(x).unsqueeze(2)
        
#         # Process each position
#         for i in range(mtp_upper):
#             try:
#                 h_prev = hidden[:, i, :].clone()
                
#                 # Generate predictions for next k tokens
#                 for k in range(config.no_head_in_mtp):
#                     future_token_idx = i + k + 1
                    
#                     # Bounds checking
#                     if future_token_idx >= seq_len:
#                         print(f"Warning: Future token index {future_token_idx} out of bounds (seq_len={seq_len})")
#                         break
                    
#                     # Get future token embedding
#                     if input_tokens is not None:
#                         # Validate token indices
#                         if torch.any(input_tokens[:, future_token_idx] < 0) or \
#                            torch.any(input_tokens[:, future_token_idx] >= config.vocab_size):
#                             print(f"Warning: Token indices out of bounds at position {future_token_idx}")
#                             future_emb = torch.zeros(batch_size, config.n_embd, 
#                                                    dtype=x.dtype, device=x.device)
#                         else:
#                             future_emb = self.token_embedding(input_tokens[:, future_token_idx])
#                     else:
#                         future_emb = x[:, future_token_idx, :]
                    
#                     # Apply normalization with numerical stability
#                     try:
#                         h_prev = self.rmsnorm(h_prev)
#                         future_emb = self.rmsnorm(future_emb)
#                     except Exception as e:
#                         print(f"Error in normalization: {e}")
#                         # Skip normalization if it fails
#                         pass
                    
#                     # Concatenate and project
#                     try:
#                         concatenated_res = torch.cat([future_emb, h_prev], dim=-1)
#                         curr_h = self.projections[k](concatenated_res)
#                     except Exception as e:
#                         print(f"Error in projection at k={k}: {e}")
#                         continue
                    
#                     # Transformer forward pass
#                     try:
#                         transformer_input = curr_h.unsqueeze(1)
#                         curr_res = self.transformer(transformer_input)
#                         curr_res = curr_res.squeeze(1)  # Remove sequence dimension
#                     except Exception as e:
#                         print(f"Error in transformer pass: {e}")
#                         curr_res = curr_h  # Use projection output directly
                    
#                     # Generate logits
#                     try:
#                         logits = self.unembd(curr_res)
                        
#                         # Validate logits
#                         if torch.any(torch.isnan(logits)) or torch.any(torch.isinf(logits)):
#                             print(f"NaN/Inf detected in logits at position [{i}, {k}]")
#                             logits = torch.zeros_like(logits)
                        
#                         # Store logits safely
#                         all_logits[:, i, k, :] = logits
                        
#                     except Exception as e:
#                         print(f"Error generating logits at position [{i}, {k}]: {e}")
#                         continue
                    
#                     # Update h_prev for next iteration
#                     h_prev = curr_h.detach()
                    
#                     # Clean up intermediate tensors
#                     del concatenated_res, curr_h, curr_res, logits
                    
#             except Exception as e:
#                 print(f"Error processing position {i}: {e}")
#                 continue
            
#             # Periodic memory cleanup
#             if i % 10 == 0 and device == 'cuda':
#                 torch.cuda.empty_cache()
        
#         # Final validation of output
#         if torch.any(torch.isnan(all_logits)) or torch.any(torch.isinf(all_logits)):
#             print("Warning: NaN/Inf detected in final output")
#             all_logits = torch.nan_to_num(all_logits, nan=0.0, posinf=1e6, neginf=-1e6)
        
#         return all_logits  
