
```markdown
# Investigation on Stability and Faithfulness in LLM Reasoning Explanations

Welcome to the backend repository for our major project! This codebase serves as our research engine to systematically stress-test Large Language Models (LLMs). We are moving away from subjective "LLM-as-a-judge" methods and using objective, XAI-based metrics to identify error patterns and quantify hallucinations.

---

## 📂 Project Structure

This repository is kept intentionally clean and modular. Here is where everything lives:

* **`main.py`**: The FastAPI controller. This handles incoming API requests, triggers the model generation, and processes the raw neural activations into a clean JSON format.
* **`model_pipeline.py`**: Our **Mechanistic Transparency** engine. It loads the LLM, manages PyTorch layer-wise hooks to capture hidden states, and facilitates causal bottleneck verification.
* **`.gitignore`**: Ensures that heavy folders like `venv/`, `.cache/`, and system logs are not accidentally uploaded to GitHub.

---

## 🚀 How to Setup the Project Locally

If this is your first time pulling the project, follow these steps to get the server running on your machine:

1. **Clone the repository:**
   ```bash
   git clone <paste-your-github-repo-link-here>
   cd llm-xai-backend

```

2. **Set up your virtual environment:**
```bash
python -m venv venv

```


3. **Activate the virtual environment:**
* **Windows:** `venv\Scripts\activate`
* **Mac/Linux:** `source venv/bin/activate`


4. **Install the required dependencies:**
```bash
pip install -r requirements.txt

```


5. **Start the local server:**
```bash
uvicorn main:app --reload

```


*Once running, you can test the API by visiting `http://127.0.0.1:8000/docs` in your browser!*

---

## 🛠️ Team Workflow (Crucial!)

To collaborate effectively without overwriting each other's work or breaking the main project, please follow this Git branching workflow every time you code:

1. **Always pull the latest updates before starting:**
```bash
git checkout main
git pull origin main

```


2. **Create a new branch for your work (DO NOT code directly on main):**
```bash
git checkout -b your-name-or-feature

```


*(Example: `git checkout -b janvi-analysis`)*
3. **Write your code, then add and commit your changes:**
```bash
git add .
git commit -m "Added XAI heatmap logic"

```


4. **Push your specific branch to GitHub:**
```bash
git push -u origin your-name-or-feature

```


5. **Merge:** Go to GitHub.com and open a **Pull Request** to safely merge your work into the `main` branch.

---

```

```
