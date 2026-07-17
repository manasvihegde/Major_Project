# LLM Reasoning Stability Analysis Pipeline

This project evaluates the robustness of Large Language Models (LLMs) by applying algorithmic prompt perturbations and comparing the model's reasoning across original and modified prompts. The pipeline integrates dataset curation, perturbation generation, comparative logging, and PostgreSQL-based experiment tracking.

---

## Project Workflow

```
curate_dataset.py
        │
        ▼
curated_reasoning_dataset.json
        │
        ├──────────────► baseline_eval.py
        │
        ▼
run_parsers.py
        │
        ▼
CuratedDatasetParser
        │
        ▼
PostgreSQL Database
        │
        ▼
comparative_logger.py
        │
        ▼
model_pipeline.py
        │
        ▼
Model Outputs + Stability Scores
```

---

## Features

- Curated reasoning dataset generation
- Baseline evaluation metrics
- Prompt perturbation generation
- Comparative reasoning analysis
- Stability score computation
- PostgreSQL experiment logging
- GPT-2 inference support
- Offline execution using curated datasets

---

## Project Structure

```
project/
│
├── baseline_eval.py
├── comparative_logger.py
├── curate_dataset.py
├── model_pipeline.py
├── perturbation_config.py
├── perturbation_engine.py
├── run_parsers.py
│
├── parsers/
│   ├── base_parser.py
│   ├── curated_dataset_parser.py
│   └── gsm8k_parser.py
│
├── db/
│   ├── connection.py
│   ├── init_db.py
│   └── schema.sql
│
├── curated_reasoning_dataset.json
└── README.md
```

---

## Database Schema

The system stores experiment information across the following tables:

- datasets
- questions
- perturbations
- run_logs
- model_outputs
- stability_scores
- xai_attribution_scores

---

## Installation

Create a virtual environment.

```bash
python -m venv venv
```

Activate it.

### Windows

```bash
venv\Scripts\activate
```

Install dependencies.

```bash
pip install -r requirements.txt
```

or install manually

```bash
pip install transformers
pip install datasets
pip install sentence-transformers
pip install psycopg2-binary
pip install python-dotenv
pip install pandas
```

---

## Running the Project

### 1. Generate Curated Dataset

```bash
python curate_dataset.py
```

Creates:

```
curated_reasoning_dataset.json
```

---

### 2. Initialize Database

```bash
python db/init_db.py
```

This recreates all PostgreSQL tables.

---

### 3. Parse Dataset

```bash
python run_parsers.py
```

The parser automatically:

- checks whether `curated_reasoning_dataset.json` exists
- uses it if available
- otherwise falls back to downloading GSM8K from Hugging Face

During parsing the system:

- registers the dataset
- inserts questions
- generates perturbations
- stores everything in PostgreSQL

---

### 4. Run Comparative Evaluation

```bash
python comparative_logger.py --num_questions 5 --model gpt2
```

This:

- loads GPT-2
- evaluates original prompts
- evaluates perturbed prompts
- computes cosine similarity
- stores outputs
- computes stability scores

---

## Outputs

### Dataset

```
curated_reasoning_dataset.json
```

Contains curated GSM8K and StrategyQA reasoning prompts.

---

### Baseline Metrics

```
baseline_metrics_log.json
```

Contains baseline evaluation statistics.

---

### Database Tables

Questions

```sql
SELECT COUNT(*) FROM questions;
```

Perturbations

```sql
SELECT COUNT(*) FROM perturbations;
```

Model Outputs

```sql
SELECT COUNT(*) FROM model_outputs;
```

Stability Scores

```sql
SELECT COUNT(*) FROM stability_scores;
```

---

## Example Results

Example run using 5 questions:

```
Questions inserted      : 6
Perturbations generated : 24
Model outputs logged    : 25
Stability scores        : 5
```

---

## Technologies Used

- Python
- Hugging Face Transformers
- Hugging Face Datasets
- Sentence Transformers
- PostgreSQL
- Psycopg2
- Pandas
- GPT-2

---


## Future Improvements

- Support additional reasoning datasets
- Multi-model benchmarking
- Advanced perturbation strategies
- XAI visualization dashboard
- Web interface for experiment monitoring
