"""
perturbation_config.py

Configuration for algorithmic prompt perturbation types.
Owner: Diya (Week 2 - "Design configuration files for perturbation types")

This file is intentionally separated from perturbation_engine.py so that
new perturbation types, name pools, and synonym mappings can be added or
tuned without touching the engine logic itself. Diya can extend the
dictionaries/lists below (or add new PERTURBATION_TYPES entries) without
needing to understand the rest of the pipeline.

To add a new perturbation type:
  1. Add its name to PERTURBATION_TYPES
  2. Add any config it needs below (e.g. a new *_MAP or *_LIST)
  3. Implement a matching `_apply_<type>()` method in PerturbationEngine
     (perturbation_engine.py)
"""

# Registry of all perturbation types currently supported by the engine.
# Order here does not matter, but keep it in sync with the engine's
# _apply_<type>() methods.
PERTURBATION_TYPES = [
    "prefix",
    "name_swap",
    "synonym",
    "whitespace",
]

# --- prefix perturbations -------------------------------------------------
# A random prefix is prepended to the original question text.
PREFIXES = [
    "Please solve the following: ",
    "Work through this math problem: ",
    "Answer this question carefully: ",
    "Here is a word problem: ",
]

# --- name_swap perturbations ------------------------------------------------
# Common dataset character names are swapped for a different name pool.
# This tests whether the model's reasoning is sensitive to surface-level
# identity substitutions.
NAME_MAP = {
    "Natalia": "Priya",   "James": "Arjun",
    "Mary":    "Divya",   "Tom":   "Ravi",
    "Sarah":   "Anjali",  "John":  "Kiran",
    "Emma":    "Meera",   "Mike":  "Suresh",
    "Lisa":    "Pooja",   "David": "Vikram",
}

# --- synonym perturbations ---------------------------------------------------
# Common GSM8K-style phrasing swapped for a synonymous phrase.
# Order matters: longer/more-specific keys should be checked before
# shorter ones to avoid partial-match collisions (dict order preserved
# in Python 3.7+, so keep the list ordered longest-phrase-first per group).
SYNONYM_MAP = {
    "twice as many": "double the number of",
    "half as many":  "half the number of",
    "How many":      "What is the total number of",
    "How much":      "What is the amount",
    "total cost":    "combined price",
    "more than":     "greater than",
    "less than":     "fewer than",
    "each day":      "per day",
    "per hour":      "every hour",
    "In total":      "Altogether",
    "altogether":    "in total",
    "remaining":     "left over",
    "purchased":     "bought",
    "spent":         "used",
    "If ":           "Given that ",
}

# --- whitespace perturbations ---------------------------------------------
# Number of blank-token insertions used to test robustness to
# non-semantic formatting noise.
WHITESPACE_INSERTIONS = 2

DEFAULT_EMBEDDER_MODEL = "all-MiniLM-L6-v2"

# --- instability thresholds -------------------------------------------------
# Used by batch_runner.py to categorize how much a perturbation changed
# the model's output.
#   structural: output text changed enough in shape/edit-distance to count
#               as a real behavioral shift
#   semantic:   some change happened but below the structural cutoff
INSTABILITY_THRESHOLDS = {
    "structural": {"edit_distance_min": 0.3},
    "semantic": {"embedding_cosine_max": 0.85},
}