

# 🚀 LLM XAI Backend - Week 2 & 3 Update

Hey team! Just a quick heads-up: **Week 2 and Week 3 tasks are officially completed and pushed.** The `main` branch is now our default, fully-merged, conflict-free branch.

If you are working on Track 3 (Data Architecture) or Track 4 (Truth-Score Evaluation), all the backend mechanics and mathematical data you need are ready to go!

## 🎯 What Was Completed

1. **Algorithmic Perturbation Engine:** Automated script to apply structural changes to prompts.
2. **Logit & Mask Extraction:** Upgraded the Hugging Face pipeline to expose token-level probabilities.
3. **Attribution Patching (Forward/Backward Hooks):** Successfully implemented gradient and activation extraction across transformer layers to track exactly *how* the LLM reasons.

---

## 🧠 How the Code Works

We added some heavy PyTorch modifications to open up the LLM's "brain." Here is a breakdown of the three main files and what they do:

### 1. `perturbation_engine.py` (The Modifier)

This acts as our algorithmic stress-tester. It takes a clean baseline prompt and applies structural perturbations (like adding spaces, swapping names, or changing casing). It pulls its configuration from `perturbation_config.py` so we can easily swap out synonyms or prefixes without breaking the code.

### 2. `model_pipeline.py` (The PyTorch Engine)

This is where the magic happens. We bypassed standard generation to manually extract the math:

* **Logits:** We added `output_logits=True` to the generation call so we can see the exact probability scores the model assigned to every word.
* **Forward Hooks:** We used `.register_forward_hook()` on the transformer layers to capture the hidden state **activations** (the model's thought process).
* **Backward Hooks (Week 3):** We added a custom `run_attribution_patching()` method that triggers a `.backward()` pass. It uses `.register_full_backward_hook()` to capture the **gradients**, showing us exactly which layers heavily influenced the final output word.

### 3. `main.py` (The API)

This exposes all our logic via FastAPI. There are three endpoints available for your scripts to hit:

* `POST /generate`: Returns standard text generation + forward activations.
* `POST /stress-test`: Runs a baseline prompt + 4 perturbed prompts, returning a comparative dictionary of logits and activations for all of them.
* `POST /attribution`: Runs a specialized forward/backward pass, returning the final predicted word, forward activations, AND backward gradients.

---

## ⚠️ Important Testing Warning!

Because the `/stress-test` and `/attribution` endpoints return massive PyTorch tensors (converted to standard JSON numbers), **do not try to test them in your web browser or the `/docs` Swagger UI.** The payload is so large it will crash Google Chrome's memory.

**How to test safely:**
If you want to pull data from these endpoints, use a Python script with the `requests` library and save the output directly to a `.json` file on your hard drive.

*Example:*

```python
import requests
import json

response = requests.post("http://127.0.0.1:8000/attribution", json={"prompt": "The capital of France is"})
with open("output.json", "w") as file:
    json.dump(response.json(), file, indent=4)

```

## ⏭️ Next Steps

Please pull from `main` to get this latest code into your local setups! You can now confidently expect the `{"prompt", "generated_text", "logits", "activations", "gradients"}` data structure from the API to start building out the logging and evaluation scripts.

Let me know if you run into any issues calling the endpoints!
