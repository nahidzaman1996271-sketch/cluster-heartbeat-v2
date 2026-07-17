"""Step 5-8 — FingerprintNet: the multi-head heartbeat model.

A shared MLP encoder compresses a telemetry window ``(F x T)`` into a compact
**workload heartbeat embedding** ``z``. Four heads consume ``z``:

* decoder       → reconstructs the window      (self-supervised anomaly signal)
* classifier    → workload behavior class      (scheduling context)
* demand head   → next-step GPU/mem/power      (placement & capacity planning)
* TTF head      → log1p(hours to failure)      (predictive ops)

The unified fingerprint returned at inference is
``[z | anomaly | class probs | demand | ttf]`` — one vector per node per
window that powers all three downstream services.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _mlp(dims: list[int], dropout: float, final_activation: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2 or final_activation:
            layers += [nn.LayerNorm(dims[i + 1]), nn.GELU(), nn.Dropout(dropout)]
    return nn.Sequential(*layers)


class FingerprintNet(nn.Module):
    """Multi-head autoencoder producing the unified workload fingerprint."""

    def __init__(
        self,
        n_features: int,
        window_size: int,
        latent_dim: int = 32,
        hidden_dims: list[int] | None = None,
        num_classes: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [256, 128]
        self.n_features = n_features
        self.window_size = window_size
        self.latent_dim = latent_dim
        input_dim = n_features * window_size

        self.encoder = _mlp([input_dim, *hidden_dims, latent_dim], dropout)
        self.decoder = _mlp([latent_dim, *hidden_dims[::-1], input_dim], dropout)
        self.classifier = _mlp([latent_dim, 64, num_classes], dropout)
        self.demand_head = _mlp([latent_dim, 64, 3], dropout)   # gpu, mem, power (scaled)
        self.ttf_head = _mlp([latent_dim, 64, 1], dropout)      # log1p(hours)

    # -- API ---------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """x: (B, F, T) normalized windows."""
        flat = x.flatten(start_dim=1)
        z = self.encoder(flat)
        return {
            "recon": self.decoder(z).view(-1, self.n_features, self.window_size),
            "latent": z,
            "cls_logits": self.classifier(z),
            "demand": self.demand_head(z),
            "ttf": self.ttf_head(z),
        }

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x.flatten(start_dim=1))

    @torch.no_grad()
    def fingerprint(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Inference-mode forward returning everything the pipeline needs."""
        self.eval()
        out = self.forward(x)
        out["cls_probs"] = torch.softmax(out["cls_logits"], dim=-1)
        return out


def reconstruction_errors(
    recon: torch.Tensor, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-feature and aggregated squared reconstruction error.

    Returns:
        (per_feature (B, F), aggregate (B,))
    """
    per_feature = ((recon - x) ** 2).mean(dim=2)
    return per_feature, per_feature.mean(dim=1)
