from MLA import MLA
from MTP import MTP
from MOE import MOE
from MLA import Config
import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
from transformers import GPT2Tokenizer


config = Config()
tokenizer = GPT2Tokenizer.from_pretrained(config.model_name)
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

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
        self.lm_head.single_token_head.weight = self.transformer.wte.weight
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
    

class Predictor: 
    def __init__(self, model_path):
        self.info = torch.load(model_path, weights_only=False)
        self.config = self.info['config']
        self.model = Deepseek()
        self.model.load_state_dict(self.info['model_state_dict'])
        self.model.eval()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.config.model_name)
    def predict(self, text, max_new_tokens=50):
        """Efficient prediction using KV cache"""
        tokens = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        initial_length = tokens.size(-1)
        
        # Initialize KV caches as None for the first forward pass
        kr_caches = None
        starting_pos = 0
        
        # Process the initial prompt
        with torch.no_grad():
            logits, _, kr_caches = self.model(tokens, inference_mode=True, kr_caches=kr_caches, starting_pos=starting_pos)
            
            next_token_logits = logits[:, -1, :]
            
            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(topk_probs, num_samples=1)
            next_token = torch.gather(topk_indices, -1, ix)
            
            # Add the new token
            tokens = torch.cat([tokens, next_token], dim=-1)
            starting_pos = initial_length
        
        # Generate remaining tokens one by one using cache
        for step in range(max_new_tokens - 1):
            with torch.no_grad():
                # Only process the last token (new token)
                last_token = tokens[:, -1:] 
                
                # Forward pass with KV cache
                logits, _, kr_caches = self.model(
                    last_token, 
                    inference_mode=True, 
                    kr_caches=kr_caches, 
                    starting_pos=starting_pos + step
                )
                
                next_token_logits = logits[:, -1, :]
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                ix = torch.multinomial(topk_probs, num_samples=1)
                next_token = torch.gather(topk_indices, -1, ix)
                
                # Add to sequence
                tokens = torch.cat([tokens, next_token], dim=-1)
                
                # Stop if we hit EOS token
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        # Decode and return
        for i in range(tokens.size(0)):
            print(f"Sequence {i + 1}: {self.tokenizer.decode(tokens[i].tolist())}")
        
        return tokens
    def predict_without_cache(self, text, max_new_tokens=50):
        """Original prediction without KV cache (for comparison)"""
        tokens = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        initial_token_size = tokens.size(-1)

        while tokens.size(-1) < initial_token_size + max_new_tokens:
            with torch.no_grad():
                logits, _, _ = self.model(tokens, inference_mode=True)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, -1)
                topk_probs, topk_indices = torch.topk(probs, 50, -1)
                ix = torch.multinomial(input=topk_probs, num_samples=1)
                ix = torch.gather(topk_indices, -1, ix)
                tokens = torch.cat((tokens, ix), -1)

        for i in range(tokens.size(0)):
            print(f"Sequence {i + 1}: {self.tokenizer.decode(tokens[i].tolist())}")
        
        return tokens
    
predictor = Predictor('trained_model_deepseek.pth')
# with open("wikipedia_2percent.txt", "r", encoding="utf-8") as f:
#     full_text = f.read()
# print(f"Now prediction is starting------")
predictor.predict('Once upon a time')
# tokens_no_cache = predictor.predict_without_cache('Anarchism is', max_new_tokens=100)
    
