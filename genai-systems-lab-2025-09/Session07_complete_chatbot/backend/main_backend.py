# pip install fastapi uvicorn python-dotenv openai
# uvicorn main_backend:app --reload



# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
import os

# --- Load Environment Variables ---
load_dotenv(override=True, dotenv_path="../../.env")
my_api_key = os.getenv("OPENAI_API_KEY")

if not my_api_key:
    raise ValueError("Missing OPENAI_API_KEY in environment variables.")

# --- Initialize OpenAI Client ---
client = OpenAI(api_key=my_api_key)

# --- Initialize FastAPI App ---
app = FastAPI(
    title="Fine-tuned OpenAI Backend",
    description="API to query fine-tuned GPT-4 model",
    version="1.0.0"
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Request Schema ---
class QuestionRequest(BaseModel):
    prompt: str

# --- Endpoint ---
@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """
    Send a user prompt to the fine-tuned model and return the response.
    """
    try:
        print(f"User asked: {request.prompt}")
        response = client.chat.completions.create(
            model="ft:gpt-4.1-2025-04-14:personal:brand-voice-support:CPjcskrj",  # Replace with your fine-tuned model
            messages=[
                {"role": "system", "content": "You are a Real Estate Agent Assistant. Answer clearly so the real estate agent can do their job meanigfully."},
                {"role": "user", "content": request.prompt}
            ]
        )
        answer = response.choices[0].message.content
        return {"response": answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Root Endpoint ---
@app.get("/")
def root():
    return {"message": "Fine-tuned GPT-4 API is running 🚀"}
