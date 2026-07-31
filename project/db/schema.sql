-- ============================================================
-- LLM Reasoning Logs — Full Schema
-- ============================================================

-- Drop tables in reverse dependency order (for clean resets)
DROP TABLE IF EXISTS instability_categories CASCADE;
DROP TABLE IF EXISTS xai_attribution_scores CASCADE;
DROP TABLE IF EXISTS stability_scores CASCADE;
DROP TABLE IF EXISTS model_outputs CASCADE;
DROP TABLE IF EXISTS perturbations CASCADE;
DROP TABLE IF EXISTS questions CASCADE;
DROP TABLE IF EXISTS datasets CASCADE;
DROP TABLE IF EXISTS run_logs CASCADE;

-- ============================================================
-- TABLE 1: datasets
-- Tracks which dataset each question came from.
-- ============================================================
CREATE TABLE datasets (
    id          SERIAL PRIMARY KEY,
    name        VARCHAR(50) NOT NULL UNIQUE,  -- 'gsm8k', 'hotpotqa', 'strategyqa'
    description TEXT,
    split       VARCHAR(20),                  -- 'train', 'test', 'validation'
    total_samples INT,
    loaded_at   TIMESTAMP DEFAULT NOW()
);


-- ============================================================
-- TABLE 2: questions
-- One row per original question from the dataset.
-- ============================================================
CREATE TABLE questions (
    id                  SERIAL PRIMARY KEY,
    dataset_id          INT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    original_index      INT NOT NULL,          -- index in the original dataset
    question_text       TEXT NOT NULL,
    ground_truth_answer TEXT NOT NULL,
    answer_numeric      FLOAT,                 -- extracted number (for GSM8K)
    question_type       VARCHAR(50),           -- 'math', 'multi_hop', 'boolean'
    num_sentences       INT,                   -- length of question
    created_at          TIMESTAMP DEFAULT NOW(),

    UNIQUE(dataset_id, original_index)
);


-- ============================================================
-- TABLE 3: perturbations
-- One row per variant of a question.
-- ============================================================
CREATE TABLE perturbations (
    id                  SERIAL PRIMARY KEY,
    question_id         INT NOT NULL REFERENCES questions(id) ON DELETE CASCADE,
    perturbation_type   VARCHAR(30) NOT NULL,
    perturbed_text      TEXT NOT NULL,
    diff_from_original  FLOAT,
    created_at          TIMESTAMP DEFAULT NOW(),

    UNIQUE (question_id, perturbation_type)
);


-- ============================================================
-- TABLE 4: run_logs
-- Tracks each experiment run (model + settings).
-- ============================================================
CREATE TABLE run_logs (
    id              SERIAL PRIMARY KEY,
    model_name      VARCHAR(100) NOT NULL,     -- 'gpt2', 'gpt2-medium', 'llama-2-7b'
    run_description TEXT,
    max_new_tokens  INT,
    temperature     FLOAT,
    decoding        VARCHAR(20),               -- 'greedy', 'sampling', 'beam'
    device          VARCHAR(20),               -- 'cpu', 'cuda'
    started_at      TIMESTAMP DEFAULT NOW(),
    completed_at    TIMESTAMP,
    status          VARCHAR(20) DEFAULT 'running'  -- 'running', 'done', 'failed'
);


-- ============================================================
-- TABLE 5: model_outputs
-- Stores the generated reasoning chain for each input.
-- Covers both original questions AND perturbation variants.
-- ============================================================
CREATE TABLE model_outputs (
    id                  SERIAL PRIMARY KEY,
    run_id              INT NOT NULL REFERENCES run_logs(id) ON DELETE CASCADE,
    question_id         INT NOT NULL REFERENCES questions(id) ON DELETE CASCADE,
    perturbation_id     INT REFERENCES perturbations(id) ON DELETE CASCADE,
    input_text          TEXT NOT NULL,
    reasoning_chain     TEXT,
    final_answer_raw    TEXT,
    final_answer_numeric FLOAT,
    is_correct          BOOLEAN,
    token_count         INT,
    generation_time_ms  INT,
    final_answer_probability FLOAT,     -- ADD THIS LINE (Week 4, regression_tracker.py)
    created_at          TIMESTAMP DEFAULT NOW()
);


-- ============================================================
-- TABLE 6: stability_scores
-- One row per question per run — compares original vs variants.
-- ============================================================
CREATE TABLE stability_scores (
    id                  SERIAL PRIMARY KEY,
    run_id              INT NOT NULL REFERENCES run_logs(id) ON DELETE CASCADE,
    question_id         INT NOT NULL REFERENCES questions(id) ON DELETE CASCADE,

    -- Scores per perturbation type (cosine similarity, 0 to 1)
    score_prefix        FLOAT,
    score_name_swap     FLOAT,
    score_synonym       FLOAT,
    score_whitespace    FLOAT,

    -- Aggregate
    overall_stability   FLOAT NOT NULL,
    stability_category  VARCHAR(20),           -- 'stable', 'moderate', 'unstable'

    -- Embedding method used
    embedder_model      VARCHAR(100) DEFAULT 'all-MiniLM-L6-v2',

    created_at          TIMESTAMP DEFAULT NOW(),

    UNIQUE(run_id, question_id)
);


-- ============================================================
-- TABLE 7: xai_attribution_scores
-- One row per token per question — stores XAI importance scores.
-- ============================================================
CREATE TABLE xai_attribution_scores (
    id                      SERIAL PRIMARY KEY,
    run_id                  INT NOT NULL REFERENCES run_logs(id) ON DELETE CASCADE,
    question_id             INT NOT NULL REFERENCES questions(id) ON DELETE CASCADE,
    perturbation_id         INT REFERENCES perturbations(id),

    token_index             INT NOT NULL,      -- position of the token in input
    token_text              VARCHAR(100),      -- the actual token string

    -- Attribution scores from different methods
    activation_patch_score  FLOAT,            -- from TransformerLens patching
    shap_score              FLOAT,            -- from SHAP
    lime_score              FLOAT,            -- from LIME
    integrated_grad_score   FLOAT,            -- from Captum

    -- Faithfulness metrics
    is_in_stated_explanation BOOLEAN,         -- did model mention this token?
    faithfulness_gap         FLOAT,           -- gap between stated vs actual
    truth_score              FLOAT,           -- combined final score

    xai_method              VARCHAR(50),       -- which method produced this row
    created_at              TIMESTAMP DEFAULT NOW()
);

-- ============================================================
-- TABLE 8: instability_categories
-- Diya, Week 2 - "Run batch jobs to categorize structural instability"
-- One row per (run, question, perturbation) — categorizes how much the
-- model's OUTPUT changed (structural/semantic/stable), separate from
-- the cosine-similarity-based stability_scores table.
-- ============================================================
CREATE TABLE instability_categories (
    id                   SERIAL PRIMARY KEY,
    run_id               INT NOT NULL REFERENCES run_logs(id) ON DELETE CASCADE,
    question_id          INT NOT NULL REFERENCES questions(id) ON DELETE CASCADE,
    perturbation_id      INT NOT NULL REFERENCES perturbations(id) ON DELETE CASCADE,
    perturbation_type    VARCHAR(30) NOT NULL,

    output_edit_ratio    FLOAT NOT NULL,   -- edit distance between original & perturbed OUTPUT text
    semantic_similarity  FLOAT,            -- pulled from stability_scores for cross-reference

    instability_category VARCHAR(20) NOT NULL,  -- 'structural', 'semantic', 'stable'

    created_at           TIMESTAMP DEFAULT NOW(),

    UNIQUE(run_id, question_id, perturbation_id)
);




-- ============================================================
-- INDEXES — speeds up the most common queries
-- ============================================================
CREATE INDEX idx_questions_dataset    ON questions(dataset_id);
CREATE INDEX idx_perturbations_qid    ON perturbations(question_id);
CREATE INDEX idx_model_outputs_run    ON model_outputs(run_id);
CREATE INDEX idx_model_outputs_qid    ON model_outputs(question_id);
CREATE INDEX idx_stability_run        ON stability_scores(run_id);
CREATE INDEX idx_stability_qid        ON stability_scores(question_id);
CREATE INDEX idx_xai_run              ON xai_attribution_scores(run_id);
CREATE INDEX idx_xai_qid              ON xai_attribution_scores(question_id);
CREATE INDEX idx_xai_token            ON xai_attribution_scores(token_index);
CREATE INDEX idx_instability_run  ON instability_categories(run_id);
CREATE INDEX idx_instability_qid  ON instability_categories(question_id);