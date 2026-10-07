import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model

# A.1: Model Registry for 3-way comparison
MODEL_REGISTRY = {
    "distilgpt2": "distilgpt2",
    "gpt-neo-125m": "EleutherAI/gpt-neo-125M",
    "gpt2": "gpt2",
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "tinyllama-1.1b": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}


class HookedModelPipeline:
    def __init__(self, model_key: str = "gpt2", peft_config=None, torch_dtype=torch.float32):
        # Resolve key from registry or allow direct HF identifier
        self.model_key = model_key
        self.model_name = MODEL_REGISTRY.get(model_key, model_key)
        
        print(f"Loading model: {self.model_key} ({self.model_name})...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # A.3: Load in CPU-friendly float32 precision without auth requirements
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            torch_dtype=torch_dtype
        ).to(self.device)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if peft_config is not None:
            self.model = get_peft_model(self.model, peft_config)
            print("PEFT / LoRA adapter successfully attached.")

        self.activations = {}
        self.hooks = []
        self.interventions = {}

        self._register_hooks()
        print(f"Model ({self.model_key}) and {len(self.hooks)} layer hooks successfully initialized on {self.device}!")

    def _get_activation_and_intervene(self, name: str):
        def hook(model, input, output):
            current_state = output[0] if isinstance(output, tuple) else output
            self.activations[name] = current_state.detach().cpu()

            if name in self.interventions:
                replacement = self.interventions[name]
                if replacement.shape != current_state.shape:
                    print(f"⚠️ FAIL-SAFE TRIGGERED at {name}: "
                          f"Expected {current_state.shape}, got {replacement.shape}. Aborting intervention.")
                else:
                    if isinstance(output, tuple):
                        return (replacement,) + output[1:]
                    return replacement
            return output
        return hook

    def _register_hooks(self):
        """
        A.2: Architecture compatibility requirement.
        Strictly resolves layers for GPT-2 and LLaMA families, raising an error if unsupported.
        """
        layers = None

        # GPT-2 and GPT-Neo family
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            layers = self.model.transformer.h
        # LLaMA and TinyLlama family
        elif hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layers = self.model.model.layers

        # Strict exception to avoid silent hook failures producing empty activations
        if layers is None or len(layers) == 0:
            raise NotImplementedError(
                f"Model '{self.model_name}' does not expose '.transformer.h' or '.model.layers'. "
                f"Hook registration aborted."
            )

        for i, layer in enumerate(layers):
            layer_name = f"layer_{i}"
            hook_handle = layer.register_forward_hook(self._get_activation_and_intervene(layer_name))
            self.hooks.append(hook_handle)

    def generate_with_hooks(self, prompt: str, max_new_tokens: int = 50):
        self.activations.clear()
        
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                do_sample=False,
                output_scores=False,
                return_dict_in_generate=True,
            )

        generated_ids = outputs.sequences[0]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        safe_activations = {}
        for k, v in self.activations.items():
            if isinstance(v, torch.Tensor):
                safe_activations[k] = v.detach().cpu().clone()
            else:
                safe_activations[k] = v

        return {
            "prompt": prompt,
            "generated_text": generated_text,
            "activations": safe_activations
        }

    def forward_with_grad(self, prompt: str, target_text: str):
        full_text = prompt + target_text
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs, labels=inputs["input_ids"])
        return outputs.loss, outputs.logits

    def add_intervention(self, layer_name: str, replacement_tensor: torch.Tensor):
        self.interventions[layer_name] = replacement_tensor.to(self.device)

    def clear_interventions(self):
        self.interventions.clear()

    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.clear_interventions()