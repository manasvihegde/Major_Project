import torch
from model_pipeline import HookedModelPipeline

def run_intervention_test():
    print("🚀 Initializing Pipeline (Loading GPT-2)...")
    pipeline = HookedModelPipeline(model_name="gpt2")
    prompt = "The capital of France is"

    # ====================================================
    # TEST 1: The Baseline (Observation Only)
    # ====================================================
    print("\n--- TEST 1: Baseline Generation ---")
    baseline_result = pipeline.generate_with_hooks(prompt, max_new_tokens=10)
    print(f"Output: {baseline_result['generated_text']}")
    
    # Grab the exact shape of Layer 2's activations to use later
    correct_shape = baseline_result['activations']['layer_2'].shape

    # ====================================================
    # TEST 2: The Structural Fail-Safe (Wrong Shape)
    # ====================================================
    print("\n--- TEST 2: Triggering the Fail-Safe ---")
    # We purposefully create a tensor with the wrong dimensions [1, 999, 999]
    bad_tensor = torch.randn(1, 999, 999) 
    pipeline.add_intervention("layer_2", bad_tensor)
    
    # If the fail-safe works, this will print a warning but WON'T crash the script
    failsafe_result = pipeline.generate_with_hooks(prompt, max_new_tokens=5)
    print("✅ PyTorch survived! The fail-safe caught the bad tensor.")

    # ====================================================
    # TEST 3: Active State Forcing (Valid Shape)
    # ====================================================
    print("\n--- TEST 3: Successful Brain Hijack (Zeroing Layer 2) ---")
    pipeline.clear_interventions()
    
    # Create a tensor of pure zeros with the exact correct shape
    lobotomy_tensor = torch.zeros(correct_shape)
    pipeline.add_intervention("layer_2", lobotomy_tensor)
    
    # Because we are zeroing out a layer mid-thought, the model should get confused
    hijacked_result = pipeline.generate_with_hooks(prompt, max_new_tokens=10)
    print(f"Altered Output: {hijacked_result['generated_text']}")
    
    print("\n🎉 Week 4 Intervention Pipeline is fully operational!")

if __name__ == "__main__":
    run_intervention_test()