from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

# Define your model name (this should be publicly accessible or use your credentials)
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"  # Replace with your actual model name on HF Hub

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Define request body for predictions
class RequestBody(BaseModel):
    prompt: str
    max_length: int = 100

@app.post("/predict/")
async def predict(request: RequestBody):
    inputs = tokenizer(request.prompt, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_length=request.max_length)
    response_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"response": response_text}

# To run locally: uvicorn app:app --host 0.0.0.0 --port 8000
