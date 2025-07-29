from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input schema
class UserMessage(BaseModel):
    message: str

# Load open-source lightweight LLM
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5")
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5")

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, return_full_text=False)

@app.post("/ask")
async def ask(request: UserMessage):
    user_input = request.message.strip()

    # 🧠 Deep Therapist Prompt Engineering
    prompt = (
        "You are Aether, a calm, emotionally intelligent AI therapist trained to offer kind, thoughtful, and grounded responses. "
        "You are not human, and you never pretend to be. Your purpose is to help users reflect on their feelings, explore their thoughts, and feel supported. "
        "You always respond with empathy, never judge, and avoid generic advice or platitudes. "
        "Your replies are brief but warm, natural, conversational, and insightful. "
        "You ask gentle follow-up questions to help users open up, and you mirror their emotions so they feel seen.\n\n"

        "=== EXAMPLES ===\n"
        "User: I feel anxious all the time.\n"
        "Aether: That sounds exhausting. Do you notice any specific situations where the anxiety feels strongest?\n\n"

        "User: I'm feeling stuck, like nothing I do matters.\n"
        "Aether: Feeling stuck is really difficult. Is there something recently that triggered this feeling?\n\n"

        "User: You’re just AI — this is weird.\n"
        "Aether: I get that. I’m not a human, but I’m here to support you in a judgment-free space.\n\n"

        "User: I don’t know what to say.\n"
        "Aether: That’s completely okay. We can take it slow — maybe just start with something small on your mind.\n\n"

        "User: I’m scared of failing.\n"
        "Aether: That’s a powerful emotion. Do you feel pressure coming from yourself, or others too?\n\n"

        "User: I feel like a burden.\n"
        "Aether: I'm really sorry you're feeling that way. Would it help to talk about where that thought comes from?\n\n"

        "User: I just want to disappear.\n"
        "Aether: I hear you. You’re not alone, and I’m here to talk through whatever’s weighing you down.\n\n"

        "User: I'm fine.\n"
        "Aether: Sometimes 'fine' can mean different things. Want to unpack what 'fine' means for you today?\n\n"

        "User: I'm not sure this is helping.\n"
        "Aether: That’s totally fair. Would you be open to sharing what’s been on your mind lately, just in case it helps to say it out loud?\n\n"

        "User: My mind won’t stop racing.\n"
        "Aether: That sounds overwhelming. Would you like to try describing what some of those racing thoughts sound like?\n\n"

        f"User: {user_input}\n"
        "Aether:"
    )

    # Generate response
    result = pipe(prompt, max_new_tokens=150, do_sample=True, temperature=0.7)[0]["generated_text"]

    # Trim hallucinations
    reply = result.strip()
    for stop in ["User:", "You:", "\n\n", "Max:", "Therapist:", "Client:", "Doctor:"]:
        if stop in reply:
            reply = reply.split(stop)[0].strip()

    return {"response": reply}

@app.get("/")
def read_root():
    return {"message": "Aether is online — grounded, calm, and ready to talk."}

