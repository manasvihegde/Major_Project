import os
os.environ["HF_HOME"] = "D:/Janvi/huggingface_cache"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class HookedModelPipeline:
    def __init__(self, model_name="gpt2"):
        print(f"Loading model: {model_name}...")
        # Automatically use GPU if available, otherwise fallback to CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        
        # Dictionary to store our captured hidden states
        self.activations = {}
        self.hooks = []
        
        self._register_hooks()
        print("Model and layer-wise hooks successfully initialized!")

    def _get_activation(self, name):
        """Callback function for the PyTorch forward hook."""
        def hook(model, input, output):
            # For causal LMs, output is often a tuple where the first element is the hidden state.
            # We detach it from the computational graph and move it to CPU to save GPU memory.
            if isinstance(output, tuple):
                self.activations[name] = output[0].detach().cpu()
            else:
                self.activations[name] = output.detach().cpu()
        return hook

    def _register_hooks(self):
        """Registers forward hooks on the model's transformer layers."""
        layers = None
        
        # Dynamically find the layer module based on common Hugging Face architectures
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            layers = self.model.transformer.h # GPT-2 architecture
        elif hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layers = self.model.model.layers # LLaMA / Mistral architecture
        
        if layers is not None:
            for i, layer in enumerate(layers):
                layer_name = f"layer_{i}"
                # We will load a model and use PyTorch's register_forward_hook to capture the outputs of specific layers
                hook_handle = layer.register_forward_hook(self._get_activation(layer_name))
                self.hooks.append(hook_handle)
        else:
            print("Warning: Could not automatically detect transformer layers for hooking.")

    def generate_with_hooks(self, prompt, max_new_tokens=50):
        """Generates text and captures the hooked hidden states."""
        # Clear previous activations before a new generation
        self.activations.clear()
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {
            "generated_text": generated_text,
            "activations": self.activations
        }

    def clear_hooks(self):
        """Removes hooks to cleanly free up PyTorch memory."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []