"""
model_runner.py

Week 2 - "Write core programmatic interface for perturbation engine"
         "Extract token-level logits and attention masks"
Owner: Janvi

This is NOT the same thing as perturbation_engine.py. PerturbationEngine
(perturbation_engine.py, Manasvi) only generates perturbed *text* — it
never touches the model. ModelRunner is the piece that actually loads the
model (via model_pipeline.py, Week 1) and runs inference on a prompt,
returning generated text plus token-level logits and attention weights.

Usage:
    from model_runner import ModelRunner

    runner = ModelRunner(model_name="...")
    result = runner.run(prompt_text)
    trace = runner.extract_full_trace(result)
"""

from dataclasses import dataclass
from typing import Optional
import torch
from model_pipeline import load_model  # Week 1, Janvi — confirm signature matches


@dataclass
class InferenceResult:
    prompt: str
    generated_text: str
    input_ids: torch.Tensor
    output_ids: torch.Tensor
    logits: Optional[torch.Tensor] = None
    attentions: Optional[tuple] = None


class ModelRunner:
    """Runs a prompt through the model and captures generation + internals."""

    def __init__(self, model_name: str, device: str = "cuda"):
        self.device = device
        self.model, self.tokenizer = load_model(model_name, device)
        self.model.eval()

    def run(self, prompt: str, max_new_tokens: int = 256) -> InferenceResult:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            gen_out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                output_scores=True,
                output_attentions=True,
                return_dict_in_generate=True
            )

        output_ids = gen_out.sequences
        generated_text = self.tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        return InferenceResult(
            prompt=prompt,
            generated_text=generated_text,
            input_ids=inputs["input_ids"],
            output_ids=output_ids,
            logits=torch.stack(gen_out.scores) if gen_out.scores else None,
            attentions=gen_out.attentions if hasattr(gen_out, "attentions") else None
        )

    def run_pair(self, original_prompt: str, perturbed_prompt: str, max_new_tokens: int = 256):
        """Entry point batch_runner.py calls: runs original + perturbed prompt."""
        return self.run(original_prompt, max_new_tokens), self.run(perturbed_prompt, max_new_tokens)

    # ---- extraction (Janvi's second Week 2 task) ----
    def extract_token_logits(self, result: InferenceResult, top_k: int = 5):
        if result.logits is None:
            raise ValueError("No logits found — run() must use output_scores=True")
        token_data = []
        for step, step_logits in enumerate(result.logits):
            probs = torch.softmax(step_logits[0], dim=-1)
            top_probs, top_ids = torch.topk(probs, top_k)
            token_data.append({
                "step": step,
                "top_tokens": top_ids.tolist(),
                "top_probs": top_probs.tolist()
            })
        return token_data

    def extract_attention_masks(self, result: InferenceResult, layer: int = -1):
        if not result.attentions:
            raise ValueError("No attentions found — run() must use output_attentions=True")
        last_step_attn = result.attentions[-1]
        layer_attn = last_step_attn[layer]
        avg_attn = layer_attn.mean(dim=1)
        return avg_attn.squeeze(0).cpu()

    def extract_full_trace(self, result: InferenceResult, top_k: int = 5):
        return {
            "prompt": result.prompt,
            "generated_text": result.generated_text,
            "token_logits": self.extract_token_logits(result, top_k),
            "final_attention": self.extract_attention_masks(result).tolist()
        }