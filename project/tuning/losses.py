import torch
import torch.nn.functional as F

def consistency_kl_loss(baseline_logits: torch.Tensor, perturbed_logits: torch.Tensor) -> torch.Tensor:
    """KL divergence between softmax distributions of baseline and perturbed forward passes."""
    min_len = min(baseline_logits.size(1), perturbed_logits.size(1))
    b_logits = baseline_logits[:, :min_len, :]
    p_logits = perturbed_logits[:, :min_len, :]
    
    b_probs = F.log_softmax(b_logits, dim=-1)
    p_probs = F.softmax(p_logits, dim=-1)
    return F.kl_div(b_probs, p_probs, reduction='batchmean')
def reward_weighted_nll(logits: torch.Tensor, target_ids: torch.Tensor, truth_score: float, baseline_mean_truth: float = 75.0) -> torch.Tensor:
    """Scales task cross-entropy loss by Truth-Score advantage."""
    advantage = (truth_score - baseline_mean_truth) / 100.0
    ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1))
    return advantage * ce_loss

def stability_guided_loss(task_nll, baseline_logits, perturbed_logits, target_ids, truth_score, alpha=1.0, beta=0.5):
    """
    Combines task cross-entropy loss, KL consistency loss, and reward-weighted NLL.
    loss = task_nll + alpha * consistency_kl_loss + beta * reward_weighted_nll
    """
    kl_loss = consistency_kl_loss(baseline_logits, perturbed_logits)
    reward_loss = reward_weighted_nll(task_nll, target_ids, truth_score)
    
    total_loss = task_nll + (alpha * kl_loss) + (beta * reward_loss)
    return total_loss