import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class HookedModelPipeline:
    def __init__(self, model_name: str = "gpt2", device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.activations = {}
        self.hooks = []

    def clear_hooks(self):
        """Remove all active PyTorch forward hooks and reset cached activations."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.activations.clear()

    def reset_activations(self):
        """Reset activation store without detaching hook handles."""
        self.activations.clear()

    def get_activation_hook(self, name: str):
        """Creates a forward hook to save layer activations."""
        def hook(model, input, output):
            if isinstance(output, tuple):
                self.activations[name] = output[0].detach()
            else:
                self.activations[name] = output.detach()
        return hook

    def register_layer_hooks(self):
        """Registers hooks across all transformer decoder blocks."""
        self.clear_hooks()
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            layers = self.model.transformer.h
        elif hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layers = self.model.model.layers
        else:
            raise AttributeError("Unsupported model architecture for hook registration.")

        for idx, block in enumerate(layers):
            hook_name = f"layer_{idx}"
            handle = block.register_forward_hook(self.get_activation_hook(hook_name))
            self.hooks.append(handle)

    def forward_with_hooks(self, text: str):
        """Runs a forward pass and captures intermediate activations."""
        self.activations.clear()
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        return outputs, inputs

    def generate_with_hooks(self, prompt: str, max_new_tokens: int = 50) -> str:
        """Generates text from a prompt while clearing lingering activation states."""
        self.activations.clear()
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True
        ).to(self.device)
        input_ids = inputs["input_ids"]

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)