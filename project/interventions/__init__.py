"""
interventions/

Week 4 module — embedding-level interventions and regression tracking.
Owner: Manasvi Hegde
"""

from .embedding_intervention import (
    fetch_perturbation,
    text_diff_to_token_positions,
    build_embedding_intervention,
    run_intervention_for_perturbation,
)

from .regression_tracker import (
    get_answer_probability,
    update_output_probability,
    track_regression,
    compare_runs,
)

__all__ = [
    # embedding_intervention
    "fetch_perturbation",
    "text_diff_to_token_positions",
    "build_embedding_intervention",
    "run_intervention_for_perturbation",
    # regression_tracker
    "get_answer_probability",
    "update_output_probability",
    "track_regression",
    "compare_runs",
]