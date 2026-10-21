"""
=============================================================================
TransparencyHead — Soft-Masking Module
=============================================================================
Source: soft-masked-diffusion-language-models/language/transparency_head.py

This is THE novel contribution of the SMDM paper. It computes a per-token
mixing coefficient λ based on the entropy of the model's preliminary
predictions, then creates a soft mixture input for a second forward pass.

How it works:
    1. From the model's preliminary logits (pass 1), compute negative entropy
       per position: H = -Σ p(v) log p(v)
    2. Compute λ = scale · σ(steepness · (H - centre))
       - High entropy (uncertain) → low λ → mostly keep [MASK]
       - Low entropy (confident) → high λ → mix in predicted embeddings
    3. For masked positions, create soft input:
       output = (1-λ) · embed([MASK]) + λ · Σ topk_prob · embed(topk_token)
    4. Return as sparse (indices, probs) tuple for efficient embedding lookup

The 4 learnable parameters (scale, centre, steepness, temperature) are
trained with a separate (higher) learning rate as per the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def softplus_inv_param(init: float) -> nn.Parameter:
    """
    Initialize a parameter such that softplus(raw_param) = init.
    This ensures the parameter stays positive after the softplus transform.
    """
    return nn.Parameter(
        torch.log(torch.expm1(torch.tensor(init, dtype=torch.float32)))
    )


class TransparencyHead(nn.Module):
    """
    The SMDM Transparency Head: computes per-token soft-masking.

    Learnable parameters (with constrained ranges via transforms):
        scale ∈ (0, 1)     via sigmoid
        centre ∈ (-∞, 0)   via -softplus (always negative)
        steepness ∈ (0, ∞) via softplus
        temperature ∈ (0, ∞) via softplus
    """
    def __init__(self, mask_token_id: int, mixinputs_k: int = 3,
                 init_scale: float = 0.1, init_centre: float = -0.75,
                 init_steep: float = 6.66, init_temperature: float = 1.0):
        super().__init__()
        self.mask_token_id = mask_token_id
        self.mixinputs_k = mixinputs_k

        self.raw_scale = nn.Parameter(
            torch.logit(torch.tensor(init_scale), eps=1e-6))
        self.raw_centre_neg = softplus_inv_param(-init_centre)
        self.raw_steep = softplus_inv_param(init_steep)
        self.raw_temperature = softplus_inv_param(init_temperature)
        self.epsilon = 1e-6

    @property
    def scale(self):
        """Mixing scale ∈ (0, 1) — controls max influence of predictions."""
        return torch.sigmoid(self.raw_scale)

    @property
    def centre(self):
        """Entropy threshold ∈ (-∞, 0) — where sigmoid transition happens."""
        return -F.softplus(self.raw_centre_neg) - self.epsilon

    @property
    def steepness(self):
        """Sigmoid steepness ∈ (0, ∞) — how sharp the transition is."""
        return F.softplus(self.raw_steep) + self.epsilon

    @property
    def temperature(self):
        """Softmax temperature ∈ (0, ∞) — sharpness of probability distribution."""
        return F.softplus(self.raw_temperature) + self.epsilon

    def forward(self, input_ids: torch.Tensor,
                logits_prelim: torch.Tensor) -> tuple:
        """
        Args:
            input_ids: (B, T) current token IDs (contains [MASK] tokens)
            logits_prelim: (B, T, V) logits from the preliminary pass

        Returns:
            (final_indices, final_probs): sparse representation
                final_indices: (B, T, k+1) — original token + top-k predictions
                final_probs: (B, T, k+1) — mixing weights
        """
        # 1. Compute negative entropy of preliminary predictions
        logp = F.log_softmax(logits_prelim, dim=-1)  # [B, T, V]
        p = logp.exp()  # [B, T, V]
        neg_entropy = (p * logp).sum(dim=-1)  # [B, T], higher = more confident

        # 2. Compute mixing coefficient λ, only at mask positions
        mask_positions = (input_ids == self.mask_token_id)
        lambda_tensor = self.scale * torch.sigmoid(
            self.steepness * (neg_entropy - self.centre))
        lambda_tensor = torch.where(
            mask_positions, lambda_tensor, torch.zeros_like(lambda_tensor))
        lambda_tensor = lambda_tensor.unsqueeze(-1)  # (B, T, 1)

        # 3. Get top-k predictions for masked positions (efficient sparse path)
        masked_logits = logits_prelim[mask_positions]  # (N_masked, V)
        if masked_logits.shape[0] > 0:
            topk_logits, topk_indices_m = torch.topk(
                masked_logits, k=self.mixinputs_k, dim=-1)
            topk_probs_m = torch.softmax(
                topk_logits.float(), dim=-1).to(masked_logits.dtype)

            # Scatter back to full (B, T, k) tensors
            topk_indices = torch.zeros(
                (*input_ids.shape, self.mixinputs_k),
                dtype=topk_indices_m.dtype, device=input_ids.device)
            topk_probs = torch.zeros_like(topk_indices, dtype=topk_probs_m.dtype)
            topk_indices[mask_positions] = topk_indices_m
            topk_probs[mask_positions] = topk_probs_m
        else:
            topk_indices = torch.zeros(
                (*input_ids.shape, self.mixinputs_k),
                dtype=torch.long, device=input_ids.device)
            topk_probs = torch.zeros_like(topk_indices, dtype=logits_prelim.dtype)

        # 4. Build sparse representation: [original_token, topk_1, ..., topk_k]
        #    with probabilities:           [(1-λ),         λ·p_1,    ..., λ·p_k]
        final_indices = torch.cat(
            [input_ids.unsqueeze(-1), topk_indices], dim=-1)  # (B, T, k+1)
        final_probs = torch.cat(
            [(1.0 - lambda_tensor), lambda_tensor * topk_probs], dim=-1)

        return (final_indices, final_probs)
