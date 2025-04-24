from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

app = FastAPI()

adapter_path = "/home/rdemari1/nlp_project/output/llama-3.2-3b-alpaca-lora/checkpoint-2500"
base_model_path = "meta-llama/Llama-3.2-3B"

tokenizer = AutoTokenizer.from_pretrained(base_model_path)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16
).to("cuda")

model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=True).to("cuda")
model.eval()

print("🖥 Model is loaded on:", next(model.parameters()).device)

@app.post("/generate")
async def generate(request: Request):
    print("Received /generate request")
    data = await request.json()
    prompt = data.get("prompt", "")
    print("Prompt:", prompt)

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Generated response:", response)
    return {"response": response}
