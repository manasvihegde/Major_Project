"""
interpretability/

Week 3 module — token/weight-level interpretability tools.
Owners: Manasvi Hegde, Diya Shetty, Kushi S Shetty
"""

from .tensor_normalizer import (
    normalize_tensor_weights,
    normalize_layer_weights,
    normalize_model_weights,
)

from .token_heatmap import (
    fetch_attribution_scores,
    serialize_token_importance,
    load_token_importance,
    visualize_token_heatmap,
)

from .entity_importance_review import (
    extract_key_entities,
    review_entities_for_question,
    run_entity_review,
)

from .sensitivity_variance import (
    compute_layer_activation_variance,
    compute_head_attention_variance,
    analyze_sensitivity,
)

from .plausibility_trap import (
    fetch_candidate_tokens,
    classify_trap,
    run_analysis,
)

__all__ = [
    # tensor_normalizer
    "normalize_tensor_weights",
    "normalize_layer_weights",
    "normalize_model_weights",
    # token_heatmap
    "fetch_attribution_scores",
    "serialize_token_importance",
    "load_token_importance",
    "visualize_token_heatmap",
    # entity_importance_review
    "extract_key_entities",
    "review_entities_for_question",
    "run_entity_review",
    # sensitivity_variance
    "compute_layer_activation_variance",
    "compute_head_attention_variance",
    "analyze_sensitivity",
    # plausibility_trap
   "fetch_candidate_tokens",
   "classify_trap",
   "run_analysis",
]