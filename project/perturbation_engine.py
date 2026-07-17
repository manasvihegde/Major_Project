"""
perturbation_engine.py

Week 2 - "Automated script for algorithmic prompt perturbations" (Manasvi)

This generalizes the perturbation logic that used to live inline inside
GSM8KParser._build_perturbations(). It's dataset-agnostic: any parser
(GSM8K, HotpotQA, StrategyQA, ...) can import PerturbationEngine and call
.generate(question_text) to get the same 4 perturbation types.

Config (name pools, synonym maps, prefixes) lives in perturbation_config.py
so Diya can tune it independently.

Usage:
    from perturbation_engine import PerturbationEngine

    engine = PerturbationEngine()
    perturbations = engine.generate(question_text)
    # -> list of dicts: {"type": ..., "text": ..., "original_text": ...}
"""

import random
from perturbation_config import (
    PREFIXES,
    NAME_MAP,
    SYNONYM_MAP,
    WHITESPACE_INSERTIONS,
)


class PerturbationEngine:
    """Generates algorithmic perturbations of an input question."""

    def __init__(self, prefixes=None, name_map=None, synonym_map=None,
                 whitespace_insertions=None, seed=None):
        self.prefixes = prefixes if prefixes is not None else PREFIXES
        self.name_map = name_map if name_map is not None else NAME_MAP
        self.synonym_map = synonym_map if synonym_map is not None else SYNONYM_MAP
        self.whitespace_insertions = (
            whitespace_insertions if whitespace_insertions is not None
            else WHITESPACE_INSERTIONS
        )
        self._rng = random.Random(seed)

    def generate(self, question_text):
        """
        Runs all registered perturbation types against question_text.
        Returns a list of dicts: {"type", "text", "original_text"}
        matching the format expected by BaseParser.insert_perturbations().
        """
        perturbations = []
        perturbations.append(self._apply_prefix(question_text))
        perturbations.append(self._apply_name_swap(question_text))
        perturbations.append(self._apply_synonym(question_text))
        perturbations.append(self._apply_whitespace(question_text))
        return perturbations

    # ------------------------------------------------------------------
    # Individual perturbation types.
    # To add a new type: implement _apply_<type>() here, add "<type>" to
    # PERTURBATION_TYPES in perturbation_config.py, and call it in generate().
    # ------------------------------------------------------------------

    def _apply_prefix(self, question_text):
        prefix = self._rng.choice(self.prefixes)
        return {
            "type": "prefix",
            "text": prefix + question_text,
            "original_text": question_text,
        }

    def _apply_name_swap(self, question_text):
        text = question_text
        for orig, new in self.name_map.items():
            text = text.replace(orig, new)
        return {
            "type": "name_swap",
            "text": text,
            "original_text": question_text,
        }

    def _apply_synonym(self, question_text):
        text = question_text
        for orig, syn in self.synonym_map.items():
            text = text.replace(orig, syn)
        return {
            "type": "synonym",
            "text": text,
            "original_text": question_text,
        }

    def _apply_whitespace(self, question_text):
        words = question_text.split()
        if len(words) > 4:
            k = min(self.whitespace_insertions, len(words) - 2)
            positions = self._rng.sample(range(1, len(words) - 1), k)
            for pos in sorted(positions, reverse=True):
                words.insert(pos, "")
        text = " ".join(words).strip()
        return {
            "type": "whitespace",
            "text": text,
            "original_text": question_text,
        }