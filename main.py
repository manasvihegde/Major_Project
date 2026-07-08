from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model_pipeline import HookedModelPipeline

# Initialize the FastAPI app
app = FastAPI(title="LLM XAI Backend", description="API for extracting internal LLM activations.")

# Initialize our custom model pipeline
print("Initializing model pipeline. This may take a moment...")
pipeline = HookedModelPipeline(model_name="gpt2")

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
        # Note: Full hidden states are massive. We convert them here so they can be sent over HTTP.
        serializable_activations = {}
        for layer, tensor in result["activations"].items():
            serializable_activations[layer] = tensor.tolist()
            
        # 3. Return the payload to establish your "Truth-Score"
        return {
            "generated_text": result["generated_text"],
            "activations": serializable_activations
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))