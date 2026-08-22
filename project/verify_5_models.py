import torch
from transformers import AutoModelForCausalLM

EXPANDED_REGISTRY = {
    "distilgpt2": "distilgpt2",
    "gpt-neo-125m": "EleutherAI/gpt-neo-125M",
    "gpt2": "gpt2",
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "tinyllama-1.1b": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
}

for key, hf_id in EXPANDED_REGISTRY.items():
    print(f"\nChecking {key} ({hf_id})...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            dtype=torch.float32,
            attn_implementation="eager"
        )
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            print(f"✅ SUCCESS: {key} ({len(model.transformer.h)} layers via .transformer.h)")
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            print(f"✅ SUCCESS: {key} ({len(model.model.layers)} layers via .model.layers)")
        else:
            print(f"❌ REJECT: {key} does not expose supported layer attributes.")
    except Exception as e:
        print(f"❌ LOAD ERROR on {key}: {str(e)}")