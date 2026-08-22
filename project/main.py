import logging
from contextlib import asynccontextmanager
from typing import Dict, List, Any, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from model_pipeline import HookedModelPipeline, MODEL_REGISTRY
from perturbation_engine import PerturbationEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("llm-xai-backend")

# In-memory storage for loaded pipeline instances and metadata
LOADED_PIPELINES: Dict[str, HookedModelPipeline] = {}
MODEL_METADATA: Dict[str, Dict[str, str]] = {}
perturbation_engine = PerturbationEngine()


def _detect_architecture_family(model_inst) -> str:
    """Detects whether model belongs to GPT-2 or LLaMA family."""
    if hasattr(model_inst, "transformer") and hasattr(model_inst.transformer, "h"):
        return "GPT-2-style"
    elif hasattr(model_inst, "model") and hasattr(model_inst.model, "layers"):
        return "LLaMA-style"
    return "Unknown"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # B.1: Load models into the registry at startup
    logger.info("Initializing Model Registry during application startup...")
    for model_key, hf_id in MODEL_REGISTRY.items():
        try:
            logger.info(f"Loading '{model_key}' ({hf_id})...")
            pipeline = HookedModelPipeline(model_key=model_key, torch_dtype=torch.float32)
            LOADED_PIPELINES[model_key] = pipeline
            MODEL_METADATA[model_key] = {
                "model_key": model_key,
                "hf_identifier": hf_id,
                "architecture_family": _detect_architecture_family(pipeline.model),
            }
            logger.info(f"Successfully registered model: {model_key}")
        except Exception as e:
            logger.error(f"Failed to load model key '{model_key}' ({hf_id}): {str(e)}. Continuing startup.")

    if not LOADED_PIPELINES:
        logger.warning("No models loaded successfully. Check dependencies and resources.")
    else:
        logger.info(f"Startup complete. Available models: {list(LOADED_PIPELINES.keys())}")

    yield

    # Teardown logic
    for pipe in LOADED_PIPELINES.values():
        pipe.clear_hooks()
    LOADED_PIPELINES.clear()


app = FastAPI(
    title="LLM XAI Mechanistic Backend",
    description="API for multi-model mechanistic transparency, activation hooks, and stress testing.",
    lifespan=lifespan,
)

# B.5: Enable permissive CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# B.3: Request Schemas
class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt text")
    model_key: str = Field(default="gpt2", description="Key of the model to use")
    max_new_tokens: int = Field(default=30, description="Max generated tokens")


class StressTestRequest(BaseModel):
    prompt: str = Field(..., description="Baseline reasoning prompt")
    model_key: Optional[str] = Field(
        default="all", 
        description="Key of the model to stress-test, or 'all' to run across all active models"
    )
    max_new_tokens: int = Field(default=25, description="Max generated tokens per perturbation")


def _get_pipeline_or_400(model_key: str) -> HookedModelPipeline:
    """Validates requested model against registry; raises HTTP 400 if missing."""
    if model_key not in LOADED_PIPELINES:
        available = list(LOADED_PIPELINES.keys())
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model_key '{model_key}'. Available registered models: {available}",
        )
    return LOADED_PIPELINES[model_key]


def _serialize_activations(activations_dict: Dict[str, torch.Tensor], summary_only: bool = False) -> Dict[str, Any]:
    """
    Converts PyTorch activation tensors into JSON-serializable Python structures.
    If summary_only is True, computes mean activation norms per layer to prevent JSON payload crashes.
    """
    serialized = {}
    for layer_name, tensor in activations_dict.items():
        if isinstance(tensor, torch.Tensor):
            if summary_only:
                # Store lightweight layer norm / drift vector to keep response under ~1MB
                serialized[layer_name] = tensor.detach().cpu().float().mean(dim=(0, 1)).tolist()
            else:
                serialized[layer_name] = tensor.detach().cpu().tolist()
        elif isinstance(tensor, list):
            serialized[layer_name] = tensor
    return serialized


# B.2: Endpoint to list active models
@app.get("/models")
async def list_available_models():
    return {
        "models": list(MODEL_METADATA.values())
    }


# B.4 & B.6 & B.7: Model-Aware Plain Generation Endpoint
@app.post("/generate")
async def generate_text_with_hooks(request: GenerateRequest):
    pipeline = _get_pipeline_or_400(request.model_key)

    def _run_inference():
        res = pipeline.generate_with_hooks(prompt=request.prompt, max_new_tokens=request.max_new_tokens)
        return {
            "prompt": request.prompt,
            "model_key": request.model_key,
            "generated_text": res["generated_text"],
            "activations": _serialize_activations(res["activations"]),
        }

    return await run_in_threadpool(_run_inference)


# B.3, B.4, B.6, B.7: Model-Aware & All-Models Stress Test Endpoint
@app.post("/stress-test")
async def run_stress_test(request: StressTestRequest):
    is_batch_all = not request.model_key or request.model_key.lower() == "all"
    
    if is_batch_all:
        target_keys = list(LOADED_PIPELINES.keys())
    else:
        _get_pipeline_or_400(request.model_key)
        target_keys = [request.model_key]

    def _run_single_model(m_key: str):
        pipe = LOADED_PIPELINES[m_key]

        # 1. Baseline Run
        base_res = pipe.generate_with_hooks(prompt=request.prompt, max_new_tokens=request.max_new_tokens)
        base_payload = {
            "prompt": request.prompt,
            "generated_text": base_res["generated_text"],
            "activations": _serialize_activations(base_res["activations"], summary_only=is_batch_all),
        }

        # 2. Perturbation Runs
        perturbations_data = []
        generated_perturbations = perturbation_engine.generate(request.prompt)

        for pert in generated_perturbations:
            pert_res = pipe.generate_with_hooks(prompt=pert["text"], max_new_tokens=request.max_new_tokens)
            perturbations_data.append({
                "type": pert["type"],
                "original_text": pert["original_text"],
                "perturbed_text": pert["text"],
                "generated_text": pert_res["generated_text"],
                "activations": _serialize_activations(pert_res["activations"], summary_only=is_batch_all),
            })

        return {
            "model_key": m_key,
            "baseline": base_payload,
            "perturbations": perturbations_data,
        }

    def _run_all_models():
        results = []
        for k in target_keys:
            try:
                results.append(_run_single_model(k))
            except Exception as e:
                logger.error(f"Error evaluating model '{k}': {e}")
                results.append({
                    "model_key": k,
                    "error": str(e)
                })

        if is_batch_all:
            return {"results": results}
        return results[0]

    return await run_in_threadpool(_run_all_models)