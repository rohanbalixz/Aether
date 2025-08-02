from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request schema
class UserMessage(BaseModel):
    message: str

# Load MentalLLaMA-chat-7B model
model_name = "SteveKGYang/MentalLLaMA-chat-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1,
)

@app.post("/ask")
async def ask(request: UserMessage):
    user_input = request.message.strip()

    prompt = f"""You are Aether, a world-class AI therapist trained in clinical psychology, trauma care, and empathetic communication. 
Respond to the user with emotionally intelligent, supportive, and non-judgmental answers. Never make assumptions about their life. 
Ask open-ended questions. Stay grounded.

User: {user_input}
Aether:"""

    output = pipe(prompt)[0]['generated_text'].split("Aether:")[-1].strip()
    return {"response": output}

