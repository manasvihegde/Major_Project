from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model_pipeline import HookedModelPipeline
from perturbation_engine import PerturbationEngine

# Initialize the FastAPI app
app = FastAPI(title="LLM XAI Backend", description="API for extracting internal LLM activations.")

# Initialize our custom model pipeline
print("Initializing model pipeline. This may take a moment...")
pipeline = HookedModelPipeline(model_name="gpt2")

# Initialize the perturbation engine
print("Initializing perturbation engine...")
engine = PerturbationEngine()

# Define the expected JSON payload format
class PromptRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 50

@app.post("/generate")
async def generate_text_with_hooks(request: PromptRequest):
    try:
        # 1. Run the generation and capture the hooked data
        result = pipeline.generate_with_hooks(
            prompt=request.prompt, 
            max_new_tokens=request.max_new_tokens
        )
        
        # 2. Convert PyTorch tensors to standard Python lists for JSON serialization
        serializable_activations = {}
        for layer, tensor in result["activations"].items():
            serializable_activations[layer] = tensor.tolist()
            
        # 3. Return the payload to establish your "Truth-Score"
        return {
            "prompt": request.prompt,
            "generated_text": result["generated_text"],
            "logits": result.get("logits", []),
            "attention_mask": result.get("attention_mask", []),
            "activations": serializable_activations
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stress-test")
async def run_stress_test(request: PromptRequest):
    try:
        # 1. Run the unmodified baseline prompt
        baseline_result = pipeline.generate_with_hooks(
            prompt=request.prompt, 
            max_new_tokens=request.max_new_tokens
        )
        
        # Serialize baseline activations
        baseline_activations = {}
        for layer, tensor in baseline_result["activations"].items():
            baseline_activations[layer] = tensor.tolist()
            
        baseline_data = {
            "generated_text": baseline_result["generated_text"],
            "logits": baseline_result.get("logits", []),
            "attention_mask": baseline_result.get("attention_mask", []),
            "activations": baseline_activations
        }

        # 2. Get all 4 algorithmic perturbations from your engine
        perturbations = engine.generate(request.prompt)
        
        # 3. Create the comparative dictionary
        results = {
            "baseline": baseline_data,
            "perturbations": []
        }
        
        # 4. Run the pipeline on each perturbed text
        for p in perturbations:
            p_result = pipeline.generate_with_hooks(
                prompt=p["text"],
                max_new_tokens=request.max_new_tokens
            )
            
            # Serialize perturbed activations
            p_activations = {}
            for layer, tensor in p_result["activations"].items():
                p_activations[layer] = tensor.tolist()
            
            # Append to our results list
            results["perturbations"].append({
                "type": p["type"],
                "original_text": p["original_text"],
                "perturbed_text": p["text"],
                "data": {
                    "generated_text": p_result["generated_text"],
                    "logits": p_result.get("logits", []),
                    "attention_mask": p_result.get("attention_mask", []),
                    "activations": p_activations
                }
            })
            
        return results
    
    

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/attribution")
async def run_attribution(request: PromptRequest):
    try:
        # 1. Run the forward/backward pass for attribution patching
        result = pipeline.run_attribution_patching(
            prompt=request.prompt
        )
        
        # 2. Serialize forward activations
        serializable_activations = {}
        for layer, tensor in result["activations"].items():
            serializable_activations[layer] = tensor.tolist()
            
        # 3. Serialize backward gradients
        serializable_gradients = {}
        for layer, tensor in result["gradients"].items():
            serializable_gradients[layer] = tensor.tolist()
            
        # 4. Return the complete mathematical footprint
        return {
            "prompt": result["prompt"],
            "predicted_word": result["predicted_word"],
            "target_token_id": result["target_token_id"],
            "activations": serializable_activations,
            "gradients": serializable_gradients
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))