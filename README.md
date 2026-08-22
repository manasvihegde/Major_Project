# 🔍 Investigation on Stability and Faithfulness in LLM Reasoning Explanations

A research project that stress-tests Large Language Models to measure **how stable** their reasoning is under harmless prompt changes, and **how faithful** their stated explanations are to what is actually happening inside the model. The project combines an algorithmic perturbation engine, a hooked PyTorch/HuggingFace inference pipeline, mechanistic-interpretability tooling (activation patching, embedding interventions, attribution heatmaps), a PostgreSQL logging backend, a FastAPI service, a Streamlit inspector dashboard, and an XAI-guided LoRA fine-tuning loop — all unified under a single metric we call the **Truth-Score**.

> Branch analyzed: [`janvi`](https://github.com/manasvihegde/Major_Project/tree/janvi)

---

## Table of Contents

- [Motivation](#motivation)
- [Objectives](#objectives)
- [Related Work](#related-work)
- [Our Approach](#our-approach)
- [System Architecture](#system-architecture)
- [Repository Structure](#repository-structure)
- [Database Schema](#database-schema)
- [Tech Stack](#tech-stack)
- [Setup & Installation](#setup--installation)
- [Running the Pipeline](#running-the-pipeline)
- [FastAPI Backend](#fastapi-backend)
- [Streamlit Inspector Dashboard](#streamlit-inspector-dashboard)
- [Module Reference](#module-reference)
- [Metrics Explained](#metrics-explained)
- [XAI-Guided LoRA Tuning](#xai-guided-lora-tuning)
- [Known Limitations & Warnings](#known-limitations--warnings)
- [Team](#team)

---

## Motivation

LLM-generated explanations look convincing, but two problems undermine trust in them:

- **Instability** — a minor, non-semantic edit to a prompt (extra whitespace, a swapped name, a synonym, an added prefix) can send the model down a completely different reasoning path, even when the final answer stays the same.
- **Unfaithfulness ("Hallucination of Logic")** — a model can produce a correct answer while pointing to the wrong features as justification. The stated explanation is a post-hoc rationalization rather than a description of what actually drove the decision, which makes the model *appear* trustworthy without *being* trustworthy.

This project builds tooling to detect, quantify, and eventually reduce both problems.

## Objectives

1. **Stress-test LLMs systematically** to identify error patterns and quantify hallucinations across perturbation types.
2. **Use explainable AI (XAI)** — attribution patching, embedding-level interventions, geometric drift metrics — to identify the *sources* of errors and hallucinations inside the model's decision-making process.
3. **Improve LLM reliability through XAI-guided tuning** (a LoRA fine-tuning loop that penalizes reasoning instability) and evaluate the resulting gains in robustness, accuracy, and trustworthiness.

## Related Work

| Work | Relevance |
|---|---|
| **The "Unfaithfulness Gap"** (arXiv 2603.16475) | Shows a persistent gap between a model's stated reasoning and causal interventions on it — reasoning behaves as an *influential signal* rather than a true *causal bottleneck*. Motivates our causal-bottleneck verification step. |
| **Bayesian Causal Frameworks** (arXiv 2504.14150) | Uses Bayesian counterfactual models to quantify causal effects, but depends on auxiliary LLM "judges" that can introduce secondary bias. We avoid LLM judges entirely. |
| **The DRIFT Framework** (ACL 2025) | A dual-reward (task vs. rationale) fine-tuning system, but scoped to classification tasks and doesn't scale to open-ended generative reasoning. |
| **Self-Explanation Foundations** (arXiv 2402.04614) | Core inspiration for this project — defines the "Hallucination of Logic," where a model highlights tokens that *seem* convincing to a human reader but were not the actual causal drivers of its decision. |

## Our Approach

Existing evaluation methods largely fall into two traps: "LLMs judging LLMs," and shallow, text-level consistency checks. This project differentiates itself by:

- **Eliminating secondary LLM bias** — replacing subjective LLM judges with objective, mathematical XAI metrics computed directly from model internals (hidden states, gradients, logits).
- **Mechanistic transparency** — instead of relying on textual self-explanations, we use attribution patching and token-importance analysis to visualize internal decision-making as heatmaps.
- **Causal bottleneck verification** — we don't just observe correlations; we intervene on the model's internal (embedding/activation) representations and check whether the intervention causes the expected change in the final output. This tells us whether the model's "reasoning" is causally load-bearing or just decorative.
- **Mitigating the Plausibility Bias / "Plausibility Trap"** — we prioritize mathematical faithfulness (activation-patching scores) over linguistic plausibility (text that merely *sounds* like a good explanation), and cross-reference the two to catch cases where they diverge.
- **A unified "Truth-Score"** — a single 0–100 metric that formally quantifies the Unfaithfulness Gap between a model's stated logic and its internal neural activity.
- **Addressing generative complexity** — scaling evaluation beyond simple classification to complex, multi-step, open-ended reasoning (math word problems, multi-hop QA, commonsense boolean reasoning).

## System Architecture

```
┌─────────────────┐   ┌──────────────────┐   ┌─────────────────────┐
│  Dataset Layer   │──▶│  Perturbation     │──▶│  Hooked Inference    │
│ GSM8K / HotpotQA │   │  Engine (4 types) │   │  Pipeline (GPT-2 +   │
│ StrategyQA /     │   │  prefix/name/     │   │  forward+backward    │
│ curated JSON     │   │  synonym/whitespc │   │  hooks, PEFT/LoRA)   │
└─────────────────┘   └──────────────────┘   └──────────┬───────────┘
                                                          │ activations,
                                                          │ logits, gradients
                                                          ▼
   ┌───────────────────────────────────────────────────────────────────┐
   │                    PostgreSQL (llm_reasoning_logs)                 │
   │ datasets → questions → perturbations → model_outputs →             │
   │ stability_scores / xai_attribution_scores / instability_categories │
   └───────────────────────────────────────────────────────────────────┘
                          │                         │
                          ▼                         ▼
       ┌───────────────────────────┐   ┌───────────────────────────────┐
       │  Interventions module      │   │  Interpretability module       │
       │  geometric drift, Truth-    │   │  attribution patching,         │
       │  Score, embedding-level     │   │  token heatmaps, entity        │
       │  interventions, regression  │   │  importance, plausibility trap,│
       │  tracking                   │   │  sensitivity variance          │
       └───────────────┬────────────┘   └───────────────┬────────────────┘
                        │                                │
                        ▼                                ▼
              ┌───────────────────────────────────────────────┐
              │   XAI-guided LoRA tuning (tuning/) → adapters   │
              │   Consistency-KL loss + reward-weighted NLL     │
              └───────────────────────────────────────────────┘
                        │
                        ▼
       ┌────────────────────────┐        ┌───────────────────────────┐
       │  FastAPI backend        │        │  Streamlit dashboard       │
       │  (main.py) — REST API   │        │  (app.py) — interactive    │
       │  for external clients   │        │  mentor-review inspector   │
       └────────────────────────┘        └───────────────────────────┘
```

## Repository Structure

```
Major_Project/ (branch: janvi)
├── README.md
├── requirements.txt
├── .gitignore
└── project/
    ├── .env                          # DB credentials (not committed)
    │
    ├── main.py                       # FastAPI app — /generate, /stress-test, /attribution
    ├── app.py                        # Streamlit "Stability & Faithfulness Inspector" dashboard
    │
    ├── model_pipeline.py             # HookedModelPipeline — the core PyTorch engine
    ├── model_runner.py               # Thin wrapper for programmatic generation + trace extraction
    ├── perturbation_engine.py        # Applies 4 algorithmic perturbation types to a prompt
    ├── perturbation_config.py        # Config: prefixes, name/synonym maps, instability thresholds
    │
    ├── curate_dataset.py             # Builds curated_reasoning_dataset.json (math/boolean/multi_hop)
    ├── run_parsers.py                # Orchestrates all dataset parsers into Postgres
    ├── comparative_logger.py         # Runs baseline + perturbations, logs outputs & stability scores
    ├── batch_runner.py               # Categorizes structural vs. semantic vs. stable instability
    ├── evaluation_runner.py          # Reads instability_categories, surfaces the worst offenders
    ├── baseline_eval.py              # DB-driven baseline accuracy benchmarking
    ├── comprehensive_evaluator.py    # End-to-end Truth-Score evaluation (baseline vs. LoRA-tuned)
    ├── regression_tracker.py         # Top-level convenience wrapper for run-to-run comparisons
    ├── run_pipeline.py               # One-command driver for the full Week 2 + Week 3 pipeline
    ├── run_tuning_pipeline.py        # One-command driver for the LoRA tuning + evaluation pipeline
    ├── check_columns.py / check_constraints.py   # Small DB introspection utilities
    │
    ├── parsers/
    │   ├── base_parser.py            # Abstract base: DB registration + shared insert utilities
    │   ├── gsm8k_parser.py           # GSM8K (grade-school math word problems)
    │   ├── hotpotqa_parser.py        # HotpotQA (multi-hop Wikipedia QA, distractor setting)
    │   ├── strategyqa_parser.py      # StrategyQA (implicit multi-step commonsense yes/no)
    │   └── curated_dataset_parser.py # Loads the curated_reasoning_dataset.json into Postgres
    │
    ├── interventions/
    │   ├── embedding_intervention.py # Text-diff → token positions → embedding-space intervention
    │   ├── geometric_metrics.py      # Vectorized cosine/L2 layer-drift computation
    │   ├── misalignment_heatmap.py   # Layer × token cosine-distance heatmap renderer
    │   ├── regression_tracker.py     # DB-backed answer-probability regression tracking
    │   └── truth_score.py            # Unified 0–100 Truth-Score formula
    │
    ├── interpretability/
    │   ├── tensor_normalizer.py      # Normalizes weight tensors ahead of attribution work
    │   ├── attribution_patching.py   # Runs activation-patching over stored perturbations
    │   ├── token_heatmap.py          # Renders token-importance heatmaps from xai_attribution_scores
    │   ├── entity_importance_review.py # Cross-checks attribution scores against key entities
    │   ├── plausibility_trap.py      # Flags cases where stated logic ≠ causal attribution
    │   └── sensitivity_variance.py   # Ranks layers/heads by sensitivity to perturbations
    │
    ├── tuning/
    │   ├── dataset_builder.py        # Builds (baseline, perturbed, truth-score) training tuples
    │   ├── losses.py                 # Consistency-KL loss + Truth-Score-weighted NLL loss
    │   ├── train_lora.py             # LoRA fine-tuning loop (targets c_attn / c_proj)
    │   ├── compare_runs_report.py    # Base model vs. LoRA-tuned comparative report
    │   └── evaluate_tuned.py         # Evaluation entry point for a tuned checkpoint
    │
    ├── db/
    │   ├── schema.sql                # Full Postgres schema (8 tables, see below)
    │   ├── connection.py             # psycopg2 connection helpers (raw + dict cursor)
    │   └── init_db.py                # Applies schema.sql, optional destructive reset
    │
    ├── data/
    │   └── outputs/                  # Generated heatmaps, CSV reports, LoRA checkpoints (gitignored)
    │
    └── checkpoints/
        └── lora_v1/                  # Saved LoRA adapter weights from prior training runs
```

## Database Schema

The project persists every experiment to PostgreSQL (`llm_reasoning_logs`) so runs are reproducible and comparable over time. Eight tables, defined in `db/schema.sql`:

| Table | Purpose |
|---|---|
| `datasets` | One row per source dataset (`gsm8k`, `hotpotqa`, `strategyqa`, `curated_reasoning`) with split and sample count. |
| `questions` | One row per original question, with ground-truth answer, extracted numeric answer, and question type (`math` / `multi_hop` / `boolean`). |
| `perturbations` | One row per perturbed variant of a question (`prefix`, `name_swap`, `synonym`, `whitespace`), with the edit distance from the original. |
| `run_logs` | One row per experiment run — model name, decoding settings, device, status. |
| `model_outputs` | The generated reasoning chain + final answer for every (question, perturbation) pair, including correctness, timing, token count, and answer probability. |
| `stability_scores` | Per-question cosine-similarity stability score, broken down by perturbation type, with an overall `stable` / `moderate` / `unstable` category. |
| `xai_attribution_scores` | Per-token attribution scores from multiple methods (activation patching, SHAP, LIME, Integrated Gradients) plus the faithfulness gap and Truth-Score for that token. |
| `instability_categories` | Per (run, question, perturbation) categorization of *output-level* instability (`structural` / `semantic` / `stable`), derived from edit-distance thresholds. |

Indexes are defined on every foreign key used in hot-path joins (`run_id`, `question_id`, `token_index`, etc.).

## Tech Stack

| Layer | Tools |
|---|---|
| Model & interpretability | PyTorch, HuggingFace `transformers`, `accelerate`, `peft` (LoRA) |
| Perturbation & similarity | `sentence-transformers` (`all-MiniLM-L6-v2` embedder), Python `difflib` |
| API | FastAPI + Uvicorn, Pydantic |
| Dashboard | Streamlit, Matplotlib, Seaborn |
| Data | pandas, NumPy, HuggingFace `datasets` |
| Persistence | PostgreSQL via `psycopg2` |
| Base model | `gpt2` (swappable — the hook-registration code auto-detects GPT-2-style (`transformer.h`) or LLaMA-style (`model.layers`) architectures) |

## Setup & Installation

```bash
# 1. Clone and switch to the janvi branch
git clone https://github.com/manasvihegde/Major_Project.git
cd Major_Project
git checkout janvi

# 2. Create a virtual environment
python -m venv project/venv
source project/venv/bin/activate      # Windows: project\venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cd project
cat > .env <<EOF
DB_HOST=localhost
DB_PORT=5432
DB_NAME=llm_reasoning_logs
DB_USER=your_postgres_user
DB_PASSWORD=your_postgres_password
EOF

# 5. Create the Postgres database, then apply the schema
createdb llm_reasoning_logs
python db/init_db.py          # pass reset=True inside the script to force a clean rebuild
```

## Running the Pipeline

### One command, start to finish

```bash
python run_pipeline.py --num_questions 20 --model gpt2 --max_new_tokens 50
```

This chains together, in order:

1. `curate_dataset.py` *(optional, via `--curate`)* — builds `curated_reasoning_dataset.json`.
2. `run_parsers.py` *(skip via `--skip_parsing`)* — loads GSM8K / HotpotQA / StrategyQA / curated questions + their 4 perturbations each into Postgres.
3. `comparative_logger.py` — runs the model on every original question and its perturbations, logs outputs and stability scores, and prints the new `run_id`.
4. `batch_runner.py` — categorizes each (question, perturbation) pair as `structural`, `semantic`, or `stable` based on output edit-distance and embedding cosine thresholds from `perturbation_config.py`.
5. `evaluation_runner.py` — surfaces the top-N most unstable question/perturbation pairs for a run.
6. Week-3 interpretability sweep — `tensor_normalizer`, `sensitivity_variance`, `token_heatmap`, `entity_importance_review`, `plausibility_trap`, all scoped to the same `run_id`.

The script auto-captures `run_id` from `comparative_logger.py`'s stdout and forwards it to every downstream step.

### Running steps individually

```bash
python curate_dataset.py                          # optional dataset curation
python run_parsers.py                              # populate datasets/questions/perturbations
python comparative_logger.py --num_questions 20 --model gpt2
python batch_runner.py --run_id <id>
python evaluation_runner.py --run_id <id> --top_n 10
python -m interpretability.token_heatmap
python -m interpretability.plausibility_trap --run_id <id>
python baseline_eval.py --dataset gsm8k --limit 50
```

### XAI-guided LoRA tuning

```bash
python run_tuning_pipeline.py
```

Runs `tuning.train_lora` (LoRA fine-tuning with a consistency-KL + reward-weighted-NLL loss) followed by `tuning.compare_runs_report` (base model vs. tuned-model Truth-Score comparison). Adapters land in `checkpoints/` and `data/outputs/`.

## FastAPI Backend

`main.py` exposes the hooked pipeline over HTTP (`uvicorn main:app --reload`):

| Endpoint | Method | Description |
|---|---|---|
| `/generate` | `POST` | Standard generation with forward-hook activations. Body: `{"prompt": str, "max_new_tokens": int}`. Returns generated text, logits, attention mask, and per-layer activations. |
| `/stress-test` | `POST` | Runs the baseline prompt plus all 4 algorithmic perturbations, returning a comparative dictionary of logits/activations for each. |
| `/attribution` | `POST` | Runs a forward + backward pass (`run_attribution_patching`), returning the predicted token plus both forward activations and backward gradients — the full mathematical footprint used for the Truth-Score. |

> **⚠️ Important:** `/stress-test` and `/attribution` return large PyTorch tensors serialized to JSON. Do **not** test them from a browser or the Swagger `/docs` UI — the payload can crash the tab. Use a Python script with `requests` and dump the result straight to a `.json` file instead.

## Streamlit Inspector Dashboard

```bash
streamlit run app.py
```

An interactive "Project Mentor Review Dashboard" for causal bottleneck / Truth-Score verification:

- Select the model architecture and max generation tokens from the sidebar.
- Enter a reasoning prompt and click **Run Causal Intervention Analysis**.
- See the baseline output side-by-side with the output for one algorithmic perturbation.
- View the computed **Truth-Score**, average cosine drift, and average Euclidean drift as headline metrics.
- Inspect a **layer × token misalignment heatmap** (light = stable, dark red = a critical causal fracture / hallucination trigger), plus the exact text span that triggered the divergence.
- A built-in **Layer Reference Guide** in the sidebar explains what each GPT-2 layer band tends to represent (surface syntax → context synthesis → reasoning bottleneck → decision layer).

## Module Reference

### `model_pipeline.py` — `HookedModelPipeline`

The core PyTorch engine. Loads a HuggingFace causal LM (auto-detects GPT-2-style `transformer.h` or LLaMA-style `model.layers`), optionally wraps it with a PEFT/LoRA adapter, and registers a forward hook on every transformer layer that:

- Captures the layer's hidden-state **activations**.
- Supports **live intervention** — if a replacement tensor has been registered for that layer (`add_intervention`), the hook swaps it in, with a shape-mismatch fail-safe that aborts the intervention instead of crashing PyTorch.

`generate_with_hooks()` returns generated text, per-step logits, attentions, and captured activations. `forward_with_grad()` runs a gradient-enabled forward pass for loss/gradient-based attribution work.

### `perturbation_engine.py` + `perturbation_config.py`

Dataset-agnostic perturbation generator. `PerturbationEngine.generate(question_text)` returns 4 variants:

- **prefix** — prepends a random instruction-style prefix.
- **name_swap** — swaps common dataset character names for a different name pool (tests sensitivity to surface-level identity substitutions).
- **synonym** — replaces common phrasing with a synonymous phrase.
- **whitespace** — inserts non-semantic blank-token noise.

Thresholds for categorizing the resulting output shift as `structural` or `semantic` live in `perturbation_config.py` (`INSTABILITY_THRESHOLDS`).

### `interventions/`

- **`geometric_metrics.py`** — vectorized cosine-distance and L2-distance computation between two sets of layer activations (baseline vs. perturbed/intervened).
- **`truth_score.py`** — combines normalized cosine + Euclidean drift into a single 0–100 **Truth-Score** (100 = perfectly stable/faithful).
- **`embedding_intervention.py`** — converts a stored text perturbation into a token-position diff, then builds an embedding-space intervention mask ready for causal-bottleneck testing.
- **`misalignment_heatmap.py`** — renders and saves a layer × token cosine-distance heatmap PNG to `data/outputs/`.
- **`regression_tracker.py`** — computes the model's probability on its own ground-truth answer, writes it back to `model_outputs.final_answer_probability`, and flags regressions between two runs above a configurable threshold.

### `interpretability/`

- **`tensor_normalizer.py`** — L2-normalizes weight tensors ahead of attribution/heatmap work.
- **`attribution_patching.py`** — runs activation patching over stored perturbations pulled from Postgres.
- **`token_heatmap.py`** — visualizes per-token importance scores stored in `xai_attribution_scores`.
- **`entity_importance_review.py`** — cross-references attribution scores against key entities (names, numbers, proper nouns) in each question, to check whether the model's attention actually lands on what matters.
- **`plausibility_trap.py`** — flags the specific failure mode this project is named after: cases where the causal attribution score and the stated/plausible explanation diverge.
- **`sensitivity_variance.py`** — ranks transformer layers/attention heads by how much their activations shift under perturbation, identifying the most sensitive points in the network.

### `tuning/`

- **`dataset_builder.py`** — builds `(baseline_prompt, perturbed_prompt, ground_truth_answer, truth_score, instability_category)` training tuples from prior evaluation runs.
- **`losses.py`** — `consistency_kl_loss` (KL divergence between baseline and perturbed token distributions, penalizing answer drift under harmless perturbations) and a Truth-Score-reward-weighted NLL loss.
- **`train_lora.py`** — LoRA fine-tuning loop targeting `c_attn` / `c_proj` (GPT-2 attention projections), saving adapter checkpoints.
- **`compare_runs_report.py`** — runs the same prompt through the base model and the LoRA-tuned model, computes Truth-Score for both, and reports the delta (Objective 3 of the project).

## Metrics Explained

- **Cosine drift / Euclidean drift** — per-layer distance between a baseline run's hidden-state activations and a perturbed/intervened run's activations. Near 0 = the model's internal representation barely moved; near 1 (cosine) = a large internal shift.
- **Truth-Score (0–100)** — `truth_score.py` averages normalized cosine and Euclidean drift into a single misalignment penalty, then inverts it: **100 = highly stable/faithful reasoning, 0 = complete internal divergence.**
- **Instability category** (`structural` / `semantic` / `stable`) — derived from output-level text edit-distance plus embedding cosine similarity, thresholds configured in `perturbation_config.py`.
- **Faithfulness gap** — the difference between a token's *stated* importance (does the model's explanation mention it?) and its *causal* importance (activation-patching / gradient attribution score) — the mathematical signature of the "Plausibility Trap."

## XAI-Guided LoRA Tuning

The tuning loop (`tuning/train_lora.py`) attaches a small LoRA adapter (`r=8`, `alpha=16`, dropout `0.05`) to GPT-2's attention projections and trains it with a composite loss:

- **Consistency-KL loss** — penalizes the model for producing different output distributions across a baseline prompt and its perturbations, directly targeting the *instability* problem.
- **Truth-Score-weighted NLL loss** — reweights the standard language-modeling loss by how faithful a sample's reasoning was, so faithful examples get reinforced more strongly than shallow/plausible-but-wrong ones.

`tuning/compare_runs_report.py` and `comprehensive_evaluator.py` then benchmark the tuned adapter against the base model on the same prompts, reporting Truth-Score deltas — this is the empirical evidence for Objective 3 (robustness/accuracy/trustworthiness gains from XAI-guided tuning).

## Known Limitations & Warnings

- **Large response payloads:** `/stress-test` and `/attribution` return full PyTorch tensors as JSON; never open them in a browser or Swagger UI — script the request and dump straight to disk.
- **Base model is GPT-2** by default (small, fast to iterate on, but weaker reasoning than production-grade LLMs) — the hook code is architecture-aware and can be pointed at LLaMA-style models via `model_name`.
- **`sensitivity_variance.py`** operates purely in-memory (no dedicated DB table exists yet for per-layer/per-head sensitivity scores) — results are written to `data/outputs/` as JSON rather than persisted relationally.
- **DB credentials** are read from a local `.env` (gitignored) — you must create your own before running any script that touches Postgres.
- `data/outputs/` (heatmaps, CSV reports, LoRA checkpoints) is gitignored; re-running the pipeline regenerates it locally.

