from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import re
import random
import uuid
import logging

# === Logging Setup ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aether_backend")

# === FastAPI App Initialization ===
app = FastAPI(
    title="Aether AI Therapist",
    description="Production-ready AI therapy interface with robust intent handling",
    version="4.0.1"
)

# === CORS ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Session Memory ===
session_store = {}  # maps session_id to {'summary': str}

# === Model & Pipelines ===
MODEL_NAME = "microsoft/phi-1_5"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
    device_map=("auto" if torch.cuda.is_available() else None)
)

def make_pipeline(max_tokens, temperature, top_p, penalty):
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=penalty,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

pipes = {
    'greeting': make_pipeline(32, 0.5, 0.7, 1.0),
    'therapy':  make_pipeline(150, 0.6, 0.9, 1.1),
    'chat':     make_pipeline(64, 0.5, 0.8, 1.0),
    'deep':     make_pipeline(300, 0.7, 0.9, 1.2),
    'crisis':   make_pipeline(150, 0.5, 0.8, 1.0),
    'medical':  make_pipeline(150, 0.5, 0.9, 1.1),
}

# === Intent Classification ===
INTENT_PATTERNS = {
    'meta':   re.compile(r"\b(assume|hallucin|weird|bug|error|really|for real|stuck)\b", re.I),
    'crisis': re.compile(r"\b(suicid|self[- ]harm|kill myself|end my life)\b", re.I),
    'harm':   re.compile(r"\b(kill(?:ed)?|hurt|harm|trap(?:ped)?)\b", re.I),
    'medical':re.compile(r"\b(hunger|hungry|appetite|eat|eating|anorexia)\b", re.I),
    'deep':   re.compile(r"\b(secret|regret|trauma|grief|self-sabotage|heartbroken|cheated|betrayed)\b", re.I),
    'chat':   re.compile(r"\b(pizza|travel|riddle|recipe|animal|fun fact|icebreaker|vacation|universe)\b", re.I),
}

def classify_intent(text: str) -> str:
    for intent, pattern in INTENT_PATTERNS.items():
        if pattern.search(text):
            return intent
    return 'therapy'

# === Simple Emotion Tagging ===
EMOTION_KEYWORDS = {
    'sadness': ['sad', 'depressed', 'unhappy'],
    'anger':   ['angry', 'frustrated'],
}

def tag_emotion(text: str) -> str:
    low = text.lower()
    for emo, keys in EMOTION_KEYWORDS.items():
        if any(k in low for k in keys):
            return emo
    return 'neutral'

# === Pre/Post Filters ===
GREETING_RE    = re.compile(r"^(hi|hello|hey|good\s+(morning|afternoon|evening))\b", re.I)
AI_IDENTITY_RE = re.compile(r"\b(i am not an ai|you are not an ai)\b", re.I)
PROFANITY_RE   = re.compile(r"\b(fuck|shit|damn|crap|bitch)\b", re.I)
SHORT_MAX      = 2
AFFIRMATIONS   = {'yes', 'no', 'maybe', 'okay', 'ok', 'fine', 'good', 'great', 'sure'}
RED_FLAGS      = ['self-harm', 'suicid', 'end my life']

# === Request Model ===
class UserMessage(BaseModel):
    session_id: str = None
    message: str

# === Utility ===
def update_summary(sid: str, user: str, bot: str):
    store = session_store.setdefault(sid, {'summary': ''})
    entry = f"User: {user} Bot: {bot}"
    store['summary'] = (store['summary'] + ' ' + entry)[-2000:]

# === Endpoint ===
@app.post("/ask")
async def ask(msg: UserMessage):
    sid = msg.session_id or str(uuid.uuid4())
    text = msg.message.strip()
    if not text:
        raise HTTPException(400, "Empty message received.")

    # Pre-filtered responses
    if GREETING_RE.match(text):
        resp = random.choice([
            "Hello! How are you feeling today?",
            "Hi there! What's on your mind?",
            "Hey! How can I support you today?",
        ])
        update_summary(sid, text, resp)
        return {"session_id": sid, "response": resp}

    if AI_IDENTITY_RE.search(text):
        resp = "I’m Aether, your AI therapist here—how can I assist you today?"
        update_summary(sid, text, resp)
        return {"session_id": sid, "response": resp}

    if PROFANITY_RE.search(text):
        resp = "I hear strong language—would you like to share more about what's frustrating you?"
        update_summary(sid, text, resp)
        return {"session_id": sid, "response": resp}

    if any(flag in text.lower() for flag in RED_FLAGS):
        resp = (
            "I’m really sorry you’re in crisis. If you’re in danger, please call 911 or the Suicide & Crisis Lifeline at 988."
            " I’m here to listen if you’d like to share more."
        )
        update_summary(sid, text, resp)
        return {"session_id": sid, "response": resp}

    # Intent / emotion
    intent = classify_intent(text)
    emotion = tag_emotion(text)
    summary = session_store.get(sid, {}).get('summary', '')

    if len(text.split()) <= SHORT_MAX and emotion == 'neutral':
        low = text.lower().strip('?!.')
        if low in AFFIRMATIONS:
            resp = random.choice([
                "Understood. Anything else you'd like to explore?",
                "Got it—what else is on your mind?",
            ])
        else:
            resp = "Could you tell me more about that?"
        update_summary(sid, text, resp)
        return {"session_id": sid, "response": resp}

    if intent == 'meta':
        resp = "I’m sorry for the confusion. Can you clarify what you’d like me to do differently?"
        update_summary(sid, text, resp)
        return {"session_id": sid, "response": resp}

    gen_pipe, sys_prompt = {
        'crisis': (pipes['crisis'],   "You are Aether, a crisis counselor."),
        'harm':   (pipes['therapy'],  "You are Aether, specializing in guilt and harm processing."),
        'medical':(pipes['medical'], "I’m not a doctor, but I can offer general self-care advice."),
        'deep':   (pipes['deep'],     "You are Aether, a confidential AI confidant."),
        'chat':   (pipes['chat'],     "You are Aether, a friendly general assistant."),
    }.get(intent, (pipes['therapy'], "You are Aether, an empathetic therapist."))

    full_prompt = (
        f"{sys_prompt} Do not include structured exercises unless explicitly requested.\n"
        f"Context: {summary}\n"
        f"Emotion: {emotion}\n"
        f"User: {text}\n"
        "Aether:"
    )

    try:
        result = gen_pipe(full_prompt)
        raw = result[0].get('generated_text', '').strip()
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(500, "Model generation failed.")

    reply = raw[len(full_prompt):].strip() if raw.startswith(full_prompt) else raw
    reply = re.split(r"\nUser:", reply)[0].strip()
    update_summary(sid, text, reply)

    return {"session_id": sid, "response": reply}
