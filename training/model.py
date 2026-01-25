# training/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

class GaussianNoise(nn.Module):
    """
    Injects Gaussian noise into the input tensor.
    Only active during training.
    """
    def __init__(self, sigma: float = 0.01):
        super().__init__()
        self.sigma = sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.sigma != 0:
            noise = torch.randn_like(x) * self.sigma
            return x + noise
        return x

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(2, dim=-1, keepdim=True) * (x.shape[-1] ** -0.5)
        return x / (norm + self.eps) * self.weight

class MambaBlock(nn.Module):
    """ The Mamba expert block, matching the Rust implementation. """
    def __init__(self, config: Dict):
        super().__init__()
        self.d_model = config['d_model']
        self.d_state = config['d_state']
        self.d_conv = config['d_conv']
        self.d_inner = self.d_model * config['expansion_factor']

        self.in_proj = nn.Linear(self.d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=self.d_conv,
            bias=True,
            groups=self.d_inner,
            padding=self.d_conv - 1,
        )
        self.x_proj = nn.Linear(self.d_inner, self.d_inner + 2 * self.d_state, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)

        self.A_log = nn.Parameter(torch.randn(self.d_inner, self.d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # This is a simplified forward pass for a single token (B=1, L=1)
        # A full implementation would handle sequences.
        _, L, C = x.shape
        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)

        x_conv = self.conv1d(x_inner.transpose(1, 2))[:, :, :L]
        x_conv = x_conv.transpose(1, 2)
        x_activated = F.silu(x_conv)

        # SSM (S6) simplified
        A = -torch.exp(self.A_log.float())
        dt = F.softplus(self.dt_proj(x_activated))
        # For a single token, state update is simplified
        y = x_activated * self.D.unsqueeze(0)

        y = y * F.silu(z)
        return self.out_proj(y)

class MoELayer(nn.Module):
    """ Mixture of Experts layer with Top-k routing. """
    def __init__(self, config: Dict):
        super().__init__()
        self.n_experts = config['n_experts']
        self.top_k = config['top_k']

        self.router = nn.Linear(config['d_model'], self.n_experts)
        self.experts = nn.ModuleList([MambaBlock(config) for _ in range(self.n_experts)])

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        B, L, C = x.shape
        x_flat = x.view(-1, C)

        router_logits = self.router(x_flat)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)

        top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        top_k_weights /= top_k_weights.sum(dim=-1, keepdim=True)

        # Aux loss for load balancing
        # A simplified version of the one from the original paper
        expert_mask = F.one_hot(top_k_indices, self.n_experts).sum(dim=1)
        expert_load = expert_mask.float().mean(dim=0)
        aux_loss = (expert_load * expert_load).sum()

        final_output = torch.zeros_like(x_flat)

        for i in range(self.top_k):
            expert_idx = top_k_indices[:, i]
            weight = top_k_weights[:, i].unsqueeze(-1)

            # This is a simplified gather/scatter, a real implementation
            # would be more efficient for batching.
            for j in range(self.n_experts):
                mask = expert_idx == j
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[j](expert_input.unsqueeze(1))
                    final_output[mask] += (expert_output.squeeze(1) * weight[mask])

        return final_output.view(B, L, C), aux_loss

class KiyEngineV3(nn.Module):
    """ The complete MoE-Mamba model for chess evaluation. """
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config['vocab_size'], config['d_model'])
        self.noise = GaussianNoise(sigma=config.get('noise_sigma', 0.0))

        self.layers = nn.ModuleList([MoELayer(config) for _ in range(config['n_layers'])])
        self.norm = RMSNorm(config['d_model'])

        self.policy_head = nn.Linear(config['d_model'], config['vocab_size'], bias=False)
        self.value_head = nn.Sequential(
            nn.Linear(config['d_model'], 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, input_ids: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        x = self.embedding(input_ids)
        x = self.noise(x)

        total_aux_loss = 0.0
        for layer in self.layers:
            residual = x
            x = self.norm(x)
            x, aux_loss = layer(x)
            x = x + residual
            total_aux_loss += aux_loss

        x = self.norm(x)

        # We only care about the last token for policy/value
        last_token_state = x[:, -1, :]

        policy_logits = self.policy_head(last_token_state)
        value = torch.tanh(self.value_head(last_token_state))

        avg_aux_loss = total_aux_loss / self.config['n_layers']

        return policy_logits, value, avg_aux_loss
