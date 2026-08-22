import torch
import torch.nn.functional as F

def consistency_kl_loss(baseline_logits, perturbed_logits):
    """
    KL divergence between softmax(baseline) and softmax(perturbed)
    at token positions — penalizes the model for answering differently
    when only surface form changes.
    """
    min_len = min(baseline_logits.size(1), perturbed_logits.size(1))
    b_logits = baseline_logits[:, :min_len, :]
    p_logits = perturbed_logits[:, :min_len, :]
    
    b_probs = F.log_softmax(b_logits, dim=-1)
    p_probs = F.softmax(p_logits, dim=-1)
    
    kl_loss = F.kl_div(b_probs, p_probs, reduction='batchmean')
    return kl_loss

def reward_weighted_nll(logits, target_ids, truth_score, baseline_mean_truth_score=75.0):
    """
    Lightweight REINFORCE-style term using truth_score as a reward weight
    to push generation toward high-truth-score outputs.
    """
    advantage = (truth_score - baseline_mean_truth_score) / 100.0
    nll = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1))
    return advantage * nll

def stability_guided_loss(task_nll, baseline_logits, perturbed_logits, target_ids, truth_score, alpha=1.0, beta=0.5):
    """
    Combines task cross-entropy loss, KL consistency loss, and reward-weighted NLL.
    loss = task_nll + alpha * consistency_kl_loss + beta * reward_weighted_nll
    """
    kl_loss = consistency_kl_loss(baseline_logits, perturbed_logits)
    reward_loss = reward_weighted_nll(task_nll, target_ids, truth_score)
    
    total_loss = task_nll + (alpha * kl_loss) + (beta * reward_loss)
    return total_loss