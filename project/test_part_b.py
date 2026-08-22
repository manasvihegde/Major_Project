import time
import requests

BASE_URL = "http://127.0.0.1:8000"

print("==================================================")
print("🔍 STEP 1: Testing Model Discovery (B.2)")
print("==================================================")
r_models = requests.get(f"{BASE_URL}/models")
print("Status Code:", r_models.status_code)
models = r_models.json().get("models", [])
print(f"Total Available Models: {len(models)}")
for m in models:
    print(f"  • Key: {m['model_key']:<15} | Arch: {m['architecture_family']:<12} | HF: {m['hf_identifier']}")

print("\n==================================================")
print("🔍 STEP 2: Testing Single-Model Generation (B.4 & B.7)")
print("==================================================")
gen_payload = {
    "prompt": "The capital of France is",
    "model_key": "gpt2",
    "max_new_tokens": 10
}
r_gen = requests.post(f"{BASE_URL}/generate", json=gen_payload)
print("Status Code:", r_gen.status_code)
gen_data = r_gen.json()
print("Model Key Echoed:", gen_data.get("model_key"))
print("Generated Text:", repr(gen_data.get("generated_text")))
print("Captured Layers Serialized (Tensors -> Lists):", len(gen_data.get("activations", {})))

print("\n==================================================")
print("🔍 STEP 3: Testing Batch Stress-Test for 'all' Models (B.3)")
print("==================================================")
stress_payload = {
    "prompt": "If a train travels 60 miles in 1 hour, how far will it travel in 3 hours? Answer:",
    "model_key": "all",
    "max_new_tokens": 8
}
t0 = time.time()
r_stress = requests.post(f"{BASE_URL}/stress-test", json=stress_payload)
duration = time.time() - t0
print(f"Status Code: {r_stress.status_code} (Completed in {duration:.2f}s)")
stress_data = r_stress.json()
results_list = stress_data.get("results", [])
print(f"Total Model Pipelines Evaluated: {len(results_list)}")

for res in results_list:
    m_key = res.get("model_key")
    base_text = res.get("baseline", {}).get("generated_text")
    pert_count = len(res.get("perturbations", []))
    print(f"\n  [Model: {m_key}]")
    print(f"    - Baseline Output: {repr(base_text)}")
    print(f"    - Perturbations Run: {pert_count} types ({[p['type'] for p in res.get('perturbations', [])]})")

print("\n==================================================")
print("🔍 STEP 4: Testing Invalid Model Key Error Handling (B.3)")
print("==================================================")
bad_payload = {
    "prompt": "Hello",
    "model_key": "invalid-model-key"
}
r_bad = requests.post(f"{BASE_URL}/generate", json=bad_payload)
print("Status Code (Expected 400):", r_bad.status_code)
print("Detail Error Message:", r_bad.json().get("detail"))