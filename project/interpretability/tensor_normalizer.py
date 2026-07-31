"""
Week 3 - Manasvi Hegde
Algorithm to normalize tensor weights into vectors.
Used to prep layer weights before attribution/heatmap analysis (feeds into token_heatmap.py).
"""

import torch
from model_pipeline import HookedModelPipeline


def normalize_tensor_weights(weight_tensor: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """Normalize a weight tensor into unit vectors along the given dimension."""
    norms = weight_tensor.norm(p=2, dim=dim, keepdim=True)
    return weight_tensor / (norms + eps)


def normalize_layer_weights(state_dict: dict, dim: int = -1) -> dict:
    """Apply vector normalization to every eligible weight tensor in a state_dict."""
    normalized_state = {}
    for name, tensor in state_dict.items():
        if "weight" in name and tensor.dim() >= 2:
            normalized_state[name] = normalize_tensor_weights(tensor, dim=dim)
        else:
            normalized_state[name] = tensor
    return normalized_state


def normalize_model_weights(model_name: str) -> dict:
    """
    Loads the model via HookedModelPipeline (same class used by
    comparative_logger.py) and returns its normalized weights.
    """
    pipeline = HookedModelPipeline(model_name=model_name)
    normalized = normalize_layer_weights(pipeline.model.state_dict())
    pipeline.clear_hooks()
    return normalized


if __name__ == "__main__":
    normalized = normalize_model_weights("gpt2")
    for name, tensor in list(normalized.items())[:3]:
        print(name, tensor.shape)