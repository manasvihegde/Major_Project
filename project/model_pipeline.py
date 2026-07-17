import os

# Only set HF_HOME if explicitly configured via .env — don't hardcode
# a personal machine path. Falls back to the standard HF cache location
# (~/.cache/huggingface) if HF_HOME_CACHE isn't set.
if os.getenv("HF_HOME_CACHE"):
    os.environ["HF_HOME"] = os.getenv("HF_HOME_CACHE")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class HookedModelPipeline:
    def __init__(self, model_name="gpt2"):
        print(f"Loading model: {model_name}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

        self.activations = {}
        self.hooks = []

        self._register_hooks()
        print("Model and layer-wise hooks successfully initialized!")

    def _get_activation(self, name):
        def hook(model, input, output):
            if isinstance(output, tuple):
                self.activations[name] = output[0].detach().cpu()
            else:
                self.activations[name] = output.detach().cpu()
        return hook

    def _register_hooks(self):
        layers = None
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            layers = self.model.transformer.h  # GPT-2 architecture
        elif hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layers = self.model.model.layers  # LLaMA / Mistral architecture

        if layers is not None:
            for i, layer in enumerate(layers):
                layer_name = f"layer_{i}"
                hook_handle = layer.register_forward_hook(self._get_activation(layer_name))
                self.hooks.append(hook_handle)
        else:
            print("Warning: Could not automatically detect transformer layers for hooking.")

    def generate_with_hooks(self, prompt, max_new_tokens=50):
        """
        Generates text and captures hooked hidden states, plus per-token
        logits and attention masks (needed for Week 2 perturbation /
        comparative logging work).
        """
        self.activations.clear()

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                output_scores=True,
                output_attentions=True,
                return_dict_in_generate=True,
            )

        generated_ids = outputs.sequences[0]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Per-generated-token logits (one tensor per new token, vocab-sized)
        logits = [step_scores[0].detach().cpu() for step_scores in outputs.scores] \
            if outputs.scores is not None else []

        # Attention weights from the final generation step
        # (tuple of layers -> tensor [batch, heads, seq, seq])
        attentions = None
        if outputs.attentions is not None and len(outputs.attentions) > 0:
            last_step_attn = outputs.attentions[-1]
            attentions = [layer_attn[0].detach().cpu() for layer_attn in last_step_attn]

        return {
            "generated_text": generated_text,
            "activations": self.activations,
            "logits": logits,
            "attentions": attentions,
            "input_ids": inputs["input_ids"][0].detach().cpu(),
        }

    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []