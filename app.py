import os
from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

# Load Hugging Face token from environment variables
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

# Ensure token is available
if huggingface_token is None:
    raise ValueError("HUGGINGFACE_TOKEN is not set. Add it to environment variables.")

# Define model name
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

# Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=huggingface_token)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=huggingface_token, torch_dtype=torch.float16, device_map="auto")

@app.get("/")
def home():
    return {"message": "LLaMA API is running!"}

@app.post("/generate")
def generate_text(prompt: str, max_length: int = 200):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=max_length)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": response}
