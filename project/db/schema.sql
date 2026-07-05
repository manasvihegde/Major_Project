-- ============================================================
-- LLM Reasoning Logs — Full Schema
-- ============================================================

-- Drop tables in reverse dependency order (for clean resets)
DROP TABLE IF EXISTS xai_attribution_scores CASCADE;
DROP TABLE IF EXISTS stability_scores       CASCADE;
DROP TABLE IF EXISTS model_outputs          CASCADE;
DROP TABLE IF EXISTS perturbations          CASCADE;
DROP TABLE IF EXISTS questions              CASCADE;
DROP TABLE IF EXISTS datasets               CASCADE;
DROP TABLE IF EXISTS run_logs               CASCADE;


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
    perturbation_type   VARCHAR(30) NOT NULL,  -- 'prefix', 'name_swap', 'synonym', 'whitespace'
    perturbed_text      TEXT NOT NULL,
    diff_from_original  FLOAT,                 -- character-level edit distance (0 to 1)
    created_at          TIMESTAMP DEFAULT NOW()
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
                        -- NULL means this is the original (not a perturbation)
    input_text          TEXT NOT NULL,         -- exact prompt fed to the model
    reasoning_chain     TEXT,                  -- the generated reasoning steps
    final_answer_raw    TEXT,                  -- raw answer text from model
    final_answer_numeric FLOAT,               -- extracted numeric answer (if any)
    is_correct          BOOLEAN,              -- does it match ground truth?
    token_count         INT,                  -- length of generated output
    generation_time_ms  INT,                  -- how long generation took
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