import torch
import torch.nn as nn
from MLA import Config
import torch.nn.functional as F

config = Config()

class Expert(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(config.n_embd, config.n_embd * 4)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(config.n_embd * 4, config.n_embd)
        self.layer2.NANOGPT_SCALE = 1
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        return self.layer2(self.dropout(self.relu(self.layer1(x))))
    
class MoeGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.expert_selector = nn.Linear(config.n_embd, config.num_of_experts)
        self.register_buffer("bias", torch.zeros(config.num_of_experts))
        self.register_buffer("total_tokens", torch.zeros(config.num_of_experts))

    def update_bias(self, topk_probs, topk_indices):
        with torch.no_grad():
            for k in range(config.activated_experts):
                indexes, weights = topk_indices[...,k], topk_probs[...,k]
                for i in range(config.num_of_experts):
                    mask = (indexes == i).float()
                    self.total_tokens[i] += mask.sum()
            avg = self.total_tokens.mean()
            u=0.1
            signatures = avg - self.total_tokens
            self.bias.data += u * torch.sign(signatures)

    def forward(self, x):
        routing_matrix = self.expert_selector(x) + self.bias
        topk_logits, topk_indices = torch.topk(routing_matrix, k=config.activated_experts, dim=-1)
        topk_probs = F.softmax(topk_logits, dim=-1)
        if self.training: 
            self.update_bias(topk_probs, topk_indices)
        return topk_probs, topk_indices
    
class MOE(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_experts = nn.ModuleList([Expert() for _ in range(config.num_of_shared_experts)])
        self.experts = nn.ModuleList([Expert() for _ in range(config.num_of_experts)])
        self.gate = MoeGate()
    def forward(self, x):
        B, T, C = x.shape
        shared_outputs = torch.zeros_like(x)
        for exp in self.shared_experts:
            shared_outputs = shared_outputs + exp(x)
        shared_outputs /= config.num_of_shared_experts
        topk_probs, topk_indices = self.gate(x)
        experts_outputs = torch.zeros_like(x)
        for k in range(config.activated_experts):
            indexes, probs = topk_indices[...,k], topk_probs[...,k]
            for i in range(config.num_of_experts):
                mask = (indexes == i)
                if mask.any():
                    experts_outputs[mask] += self.experts[i](x[mask]) * probs[mask].unsqueeze(-1)
        return shared_outputs + experts_outputs
        

# moe = MOE()
# x = torch.randn(4, 1024, 768)
# logits = moe(x)
# print(logits.shape)