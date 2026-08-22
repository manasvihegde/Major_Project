import numpy as np

def calculate_truth_score(cosine_dist: float, euclidean_dist: float, max_expected_euclidean: float = 10.0) -> float:
    """
    Calculates a unified Truth-Score (0 to 100).
    Higher score = highly stable/faithful reasoning.
    """
    # Normalize euclidean distance to a 0-1 scale
    norm_euclidean = min(euclidean_dist / max_expected_euclidean, 1.0)
    
    # Average the two penalty metrics (0 means identical, 1 means completely diverged)
    misalignment_penalty = (cosine_dist + norm_euclidean) / 2.0
    
    # Invert to create a "Truth" score where 100 is perfect stability
    truth_score = (1.0 - misalignment_penalty) * 100
    
    return round(truth_score, 2)