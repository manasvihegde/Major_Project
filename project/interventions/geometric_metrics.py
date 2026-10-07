"""
Week 5 - Janvi Hegde & Manasvi Hegde
Mathematical module for geometric distance and internal layer drift calculations.
"""

import torch
import torch.nn.functional as F
import numpy as np


def compute_layer_drifts(base_activations: dict, pert_activations: dict) -> dict:
    """
    Calculates Cosine and Euclidean distances between baseline and perturbed hidden states.
    """
    drifts = {}
    for layer_name in base_activations.keys():
        if layer_name in pert_activations:
            b_act = base_activations[layer_name].float()
            p_act = pert_activations[layer_name].float()

            # Mean-pool across sequence dimension if 3D tensor [batch, seq_len, hidden_dim]
            b_vec = b_act.mean(dim=1).view(-1) if b_act.dim() == 3 else b_act.view(-1)
            p_vec = p_act.mean(dim=1).view(-1) if p_act.dim() == 3 else p_act.view(-1)

            min_len = min(b_vec.shape[0], p_vec.shape[0])
            b_vec = b_vec[:min_len]
            p_vec = p_vec[:min_len]

            cos_sim = F.cosine_similarity(b_vec.unsqueeze(0), p_vec.unsqueeze(0)).item()
            cos_dist = max(0.0, min(1.0, 1.0 - cos_sim))
            euc_dist = torch.norm(b_vec - p_vec, p=2).item()

            drifts[layer_name] = {
                "cosine_distance": float(cos_dist),
                "euclidean_distance": float(euc_dist),
            }
    return drifts