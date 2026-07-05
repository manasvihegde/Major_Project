from abc import ABC, abstractmethod
import psycopg2
from db.connection import get_connection
import re

class BaseParser(ABC):
    """
    All dataset parsers inherit from this.
    Handles DB connection, dataset registration, and shared utilities.
    """

    def __init__(self, dataset_name, description, split="train"):
        self.dataset_name = dataset_name
        self.description  = description
        self.split        = split
        self.conn         = get_connection()
        self.dataset_id   = None

    def register_dataset(self, total_samples):
        """
        Inserts or updates the dataset entry in the datasets table.
        Returns the dataset ID.
        """
        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO datasets (name, description, split, total_samples)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (name) DO UPDATE
                SET total_samples = EXCLUDED.total_samples,
                    loaded_at     = NOW()
            RETURNING id;
        """, (self.dataset_name, self.description, self.split, total_samples))
        self.dataset_id = cur.fetchone()[0]
        self.conn.commit()
        cur.close()
        print(f"Dataset '{self.dataset_name}' registered with ID: {self.dataset_id}")
        return self.dataset_id

    def insert_question(self, index, question_text, answer_text,
                        answer_numeric=None, question_type=None):
        """
        Inserts a single question. Returns the question row ID.
        """
        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO questions
                (dataset_id, original_index, question_text,
                 ground_truth_answer, answer_numeric,
                 question_type, num_sentences)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (dataset_id, original_index) DO UPDATE
                SET question_text = EXCLUDED.question_text
            RETURNING id;
        """, (
            self.dataset_id,
            index,
            question_text,
            answer_text,
            answer_numeric,
            question_type,
            len(re.split(r'[.!?]+', question_text))  # sentence count
        ))
        qid = cur.fetchone()[0]
        self.conn.commit()
        cur.close()
        return qid

    def insert_perturbations(self, question_id, perturbations):
        """
        Inserts all perturbation variants for a question.
        perturbations: list of dicts with keys 'type' and 'text'
        Returns list of perturbation IDs.
        """
        cur = self.conn.cursor()
        pids = []

        for p in perturbations:
            # Compute simple character-level edit distance ratio
            original_len = max(1, len(p.get("original_text", p["text"])))
            diff = self._edit_distance_ratio(
                p.get("original_text", ""), p["text"]
            )
            cur.execute("""
                INSERT INTO perturbations
                    (question_id, perturbation_type,
                     perturbed_text, diff_from_original)
                VALUES (%s, %s, %s, %s)
                RETURNING id;
            """, (question_id, p["type"], p["text"], diff))
            pids.append(cur.fetchone()[0])

        self.conn.commit()
        cur.close()
        return pids

    def _edit_distance_ratio(self, s1, s2):
        """
        Returns a 0-1 ratio of how different two strings are.
        0 = identical, 1 = completely different.
        """
        if not s1 or not s2:
            return 1.0
        # Simple character difference ratio
        longer  = max(len(s1), len(s2))
        shorter = min(len(s1), len(s2))
        overlap = sum(c1 == c2 for c1, c2 in zip(s1, s2))
        return round(1.0 - (overlap / longer), 4)

    def close(self):
        self.conn.close()

    @abstractmethod
    def parse(self, num_samples):
        """Each parser implements its own parse logic."""
        pass