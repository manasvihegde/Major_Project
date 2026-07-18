import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Only set HF_HOME if explicitly configured via .env — don't hardcode
# a personal machine path. Falls back to the standard HF cache location
# (~/.cache/huggingface) if HF_HOME_CACHE isn't set.
if os.getenv("HF_HOME_CACHE"):
    os.environ["HF_HOME"] = os.getenv("HF_HOME_CACHE")

class HookedModelPipeline:
    def __init__(self, model_name="gpt2"):
        print(f"Loading model: {model_name}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

        self.activations = {}
        self.gradients = {} # New: Storage for backward math
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

    def _get_gradient(self, name):
        def hook(module, grad_input, grad_output):
            if isinstance(grad_output, tuple) and len(grad_output) > 0:
                self.gradients[name] = grad_output[0].detach().cpu()
            elif grad_output is not None:
                self.gradients[name] = grad_output.detach().cpu()
        return hook

    def _register_hooks(self):
        layers = None
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            layers = self.model.transformer.h
        elif hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layers = self.model.model.layers

        if layers is not None:
            for i, layer in enumerate(layers):
                layer_name = f"layer_{i}"
                
                # Register the forward hook (Activations)
                fw_handle = layer.register_forward_hook(self._get_activation(layer_name))
                self.hooks.append(fw_handle)
                
                # Register the backward hook (Gradients)
                bw_handle = layer.register_full_backward_hook(self._get_gradient(layer_name))
                self.hooks.append(bw_handle)
        else:
            print("Warning: Could not automatically detect transformer layers for hooking.")

    def generate_with_hooks(self, prompt: str, max_new_tokens: int = 50):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
    
        # Run generation with math extraction flags
        outputs = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,           # Now using the dynamic variable!
            output_logits=True,                      # Extracts the logits
            return_dict_in_generate=True,            # Forces structured output
            pad_token_id=self.tokenizer.eos_token_id # Prevents open-end generation warnings
        )

        # Decode the newly generated text
        generated_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        
        # Process logits (convert the tuple of tensors to standard Python lists)
        raw_logits = [logit.squeeze().tolist() for logit in outputs.logits]
        
        # Extract the attention mask used for the input
        attention_mask = inputs["attention_mask"].squeeze().tolist()

        # Return the exact agreed-upon contract
        return {
            "prompt": prompt,
            "generated_text": generated_text,
            "logits": raw_logits,
            "attention_mask": attention_mask,
            "activations": self.activations
        }
    def run_attribution_patching(self, prompt: str):
        # 1. Clear previous math to prevent cross-contamination
        self.model.zero_grad()
        self.activations.clear()
        self.gradients.clear()

        # 2. Tokenize the input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # 3. Standard forward pass (keeping the computation graph intact)
        outputs = self.model(**inputs)

        # 4. Isolate the logits for the very last token in the sequence
        next_token_logits = outputs.logits[0, -1, :]

        # 5. Find the token ID that the model thinks is the absolute best next word
        target_token_id = torch.argmax(next_token_logits)

        # 6. Extract the single mathematical logit for that specific chosen token
        target_logit = next_token_logits[target_token_id]

        # 7. The Magic: Trigger the backward pass on that single logit
        # This sends the gradient flowing backward through your hooks!
        target_logit.backward()

        # 8. Decode the target token so you know what word it predicted
        target_word = self.tokenizer.decode(target_token_id)

        return {
            "prompt": prompt,
            "predicted_word": target_word,
            "target_token_id": target_token_id.item(),
            "activations": self.activations,
            "gradients": self.gradients
        }
    
    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []