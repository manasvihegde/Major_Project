import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if os.getenv("HF_HOME_CACHE"):
    os.environ["HF_HOME"] = os.getenv("HF_HOME_CACHE")

class HookedModelPipeline:
    def __init__(self, model_name="gpt2"):
        print(f"Loading model: {model_name}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

        self.activations = {}
        self.hooks = []
        
        # WEEK 4: Storage for active manual state forcing
        self.interventions = {} 

        self._register_hooks()
        print("Model and layer-wise hooks successfully initialized!")

    # WEEK 4: The Interface for Staging Interventions
    def add_intervention(self, layer_name: str, replacement_tensor: torch.Tensor):
        """Stages a tensor to be forced into the model at a specific layer."""
        self.interventions[layer_name] = replacement_tensor.to(self.device)

    def clear_interventions(self):
        """Removes all manual state forcing."""
        self.interventions.clear()

    # WEEK 4: Upgraded Hook with Structural Fail-Safes
    def _get_activation_and_intervene(self, name):
        def hook(model, input, output):
            # 1. Capture the original state (Observation)
            if isinstance(output, tuple):
                current_state = output[0]
            else:
                current_state = output
                
            self.activations[name] = current_state.detach().cpu()

            # 2. Apply Active Intervention if one is staged for this layer
            if name in self.interventions:
                replacement = self.interventions[name]

                # STRUCTURAL FAIL-SAFE: Verify tensor dimensions exactly match
                # If they don't, the model will crash. We catch it here.
                if replacement.shape != current_state.shape:
                    print(f"⚠️ FAIL-SAFE TRIGGERED at {name}:")
                    print(f"   Expected shape: {current_state.shape}")
                    print(f"   Injected shape: {replacement.shape}")
                    print(f"   Action: Aborting intervention to prevent PyTorch crash.")
                else:
                    # Force the internal state
                    if isinstance(output, tuple):
                        return (replacement,) + output[1:]
                    else:
                        return replacement
            
            # Return normal output if no intervention occurred
            return output
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
                # WEEK 4: Using the new intervention-aware hook
                hook_handle = layer.register_forward_hook(self._get_activation_and_intervene(layer_name))
                self.hooks.append(hook_handle)
        else:
            print("Warning: Could not automatically detect transformer layers for hooking.")

    def generate_with_hooks(self, prompt, max_new_tokens=50):
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

        logits = [step_scores[0].detach().cpu() for step_scores in outputs.scores] \
            if outputs.scores is not None else []

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
        self.clear_interventions()