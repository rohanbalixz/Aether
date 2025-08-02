# 🧠 Aether — Your Personal AI Therapist

Aether is a calming, intelligent, GenZ-friendly web therapist built to simulate emotionally-aware conversations. It blends open-source AI models and modern web development to offer a thoughtful space for self-reflection.

> ❗Note: The backend (AI logic) currently runs locally. This site will not respond without running the FastAPI server on your machine.

## 🌐 Live Frontend

[https://rohanbalixz.github.io/Aether/](https://rohanbalixz.github.io/Aether/)

---

## 📌 Features

- 🧠 Emotional support via AI-powered therapist conversations
- 🧘 Therapeutic, non-robotic tone using prompt engineering
- ⚡ FastAPI + Hugging Face Transformers backend
- 💡 Modern responsive UI with MAANG-inspired styling
- 🔐 Fully local backend = privacy-first

---

## 🚀 Local Setup (Run Backend)

### 1. Clone the Repo

```bash
git clone https://github.com/rohanbalixz/Aether.git
cd Aether
```

### 2. Setup Python Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> If you don’t have `requirements.txt`, create one:

```bash
pip install fastapi uvicorn transformers
pip freeze > requirements.txt
```

### 3. Run the Backend

```bash
uvicorn backend.app:app --reload
```

Backend starts at `http://localhost:8000/ask`

---

## ⚙️ Update Frontend to Call Local Server

Make sure `script.js` fetches from:

```javascript
fetch("http://localhost:8000/ask", {
  method: "POST",
  ...
})
```

If you're deploying backend remotely later, just update the URL.

---

## 🛠️ To Do

- [ ] Host backend (Render, Hugging Face Inference, etc.)
- [ ] Improve prompt engineering for therapy tone
- [ ] Add journaling / mood tracking
- [ ] OAuth login for session history

---

## 🛡️ Disclaimer

Aether is not a licensed medical tool. It simulates a wellness companion but should not be used in place of professional therapy.
