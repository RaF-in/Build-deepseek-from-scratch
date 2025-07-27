import torch

class RotaryPositionalEmbeddings:
    def __init__(self, block_size: int, n_embd: int, theta: int):
        self.block_size = block_size
        self.theta = theta
        self.n_embd = n_embd
    def precompute_complex_rope(self, device: str):
        theta_numerator = torch.arange(0, self.n_embd, 2, device=device).float()
        theta = 1 / (self.theta ** (theta_numerator/self.n_embd))
        positions = torch.arange(self.block_size, device=device).float()
        freqs = torch.outer(positions, theta) # (block_size, n_embd/2)
        freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_complex
    def apply_rope(self, x, freqs):
        assert x.shape[-1] % 2 == 0, "Input to RoPE must have even last dimension"
        x = x.to(torch.float32)
        x_complex = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        freqs = freqs.unsqueeze(0).unsqueeze(1)
        mul = x_complex * freqs
        x_real = torch.view_as_real(mul)
        x_out = x_real.reshape(*x.shape)
        return x_out.type_as(x)

# Test
# rope = RotaryPositionalEmbeddings(block_size=1024, n_embd=128, theta=10000)
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(f"using device = {device}")
# freqs = rope.precompute_complex_rope(device)

# # Assume input: (batch, seq_len, num_heads, head_dim)
# x = torch.randn(1, 1024, 12, 64, device=device)
# x_rope = rope.apply_rope(x, freqs)  # Apply RoPE
# print(x_rope.shape)
