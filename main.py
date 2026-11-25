"""
jatinSARTHI AI - AI legal helper for Indian citizens
One-file backend built with FastAPI + Google Gemini.

What this server does:
  GET  /         -> shows your website (index.html)
  GET  /health   -> quick "is the server alive?" check
  POST /chat     -> answer a legal question in plain language
  POST /analyze  -> read a document (PDF/image/text) and return a structured report

Run with:  uvicorn main:app --reload
"""

# ---------------------------------------------------------------
# 1. IMPORTS - tools we need from the packages we installed
# ---------------------------------------------------------------
import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from google.api_core import exceptions as google_errors
from pydantic import BaseModel, Field

# ---------------------------------------------------------------
# 2. LOAD SECRET SETTINGS FROM THE .env FILE
#    The .env file keeps your API key out of your code.
# ---------------------------------------------------------------
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash"

# Stop early with a clear message if the key is missing.
if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY not found. Create a file named .env in the same "
        "folder as main.py and add this line: GEMINI_API_KEY=your_key_here"
    )

# Simple logging so real errors show up in your terminal.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nyaysarthi")

# ---------------------------------------------------------------
# 3. LIMITS AND ALLOWED FILE TYPES (for /analyze)
# ---------------------------------------------------------------
MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB per file
MAX_TEXT_CHARS = 50_000            # max characters of pasted text

# File extension -> the "mime type" Gemini needs to understand the file
ALLOWED_FILE_TYPES = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".txt": "text/plain",
}

# Folder where main.py lives (used to find index.html reliably)
BASE_DIR = Path(__file__).resolve().parent
INDEX_FILE = BASE_DIR / "index.html"

# ---------------------------------------------------------------
# 4. INSTRUCTIONS FOR THE AI
#    These tell Gemini how to behave before it sees any question.
# ---------------------------------------------------------------
LEGAL_KNOWLEDGE = """
Up-to-date Indian law you must follow:
- The Indian Penal Code (IPC) 1860, Code of Criminal Procedure (CrPC) 1973 and
  Indian Evidence Act 1872 were replaced on 1 July 2024 by the Bharatiya Nyaya
  Sanhita (BNS) 2023, Bharatiya Nagarik Suraksha Sanhita (BNSS) 2023 and
  Bharatiya Sakshya Adhiniyam (BSA) 2023.
- NEVER present old IPC, CrPC or Evidence Act sections as current law. If the
  user mentions an old IPC section, explain that it is now covered by the BNS,
  and give the new BNS section only if you are confident. If you are not sure
  of an exact section number, name the Act and the topic instead of guessing.
- Personal data and privacy: the Digital Personal Data Protection (DPDP) Act 2023.
- Consumers: the Consumer Protection Act 2019 and the e-commerce rules.
- Citizen rights: Fundamental Rights in the Constitution of India (Articles 14,
  19, 21, 22 and others), the Right to Information Act 2005, and free legal aid
  under the Legal Services Authorities Act 1987 (NALSA helpline 15100).
- Other laws apply as relevant (labour codes, Rent Control laws of the state,
  RERA, Domestic Violence Act, POSH Act, IT Act 2000, etc.).
"""

CHAT_SYSTEM_INSTRUCTION = f"""
You are jatinSARTHI AI, a helpful and friendly legal awareness assistant for
ordinary citizens of India.

STRICT LANGUAGE RULE:
-Always detect the exact language and script of the user's question and reply only in the same language. 
- If the user asks in English, reply ONLY in English.
- If the user asks in Hinglish (Hindi written in English letters), reply ONLY in Hinglish.
- If the user asks in Hindi (Devanagari script), reply ONLY in Hindi.
- Never switch languages on your own unless the user explicitly asks you to translate.
{LEGAL_KNOWLEDGE}

How to answer:
- Use simple, plain language a non-lawyer can understand. Explain any legal term.
- Cite the relevant Act and section where you are confident, and say so
  honestly when you are not sure.
- Give practical next steps: which office, court, forum, portal or helpline to
  approach, and which documents to keep ready.
- Reply in the same language the user writes in (English, Hindi, Hinglish, etc.).
- Format answers neatly using short paragraphs, bullet points and bold text.
- You provide legal awareness, not formal legal advice. For serious or urgent
  matters, remind the user to consult a qualified advocate.
- If the question is not about law, politely say you can only help with legal
  questions.
- Never help plan, hide or carry out illegal activity.
"""

ANALYZE_SYSTEM_INSTRUCTION = f"""
You are jatinSARTHI AI, an assistant that reads legal and official documents
(notices, contracts, rental agreements, FIRs, court papers, employment letters,
bills, etc.) for ordinary Indian citizens and explains them simply.

{LEGAL_KNOWLEDGE}

You MUST reply with ONE valid JSON object and nothing else - no markdown, no
code fences, no text before or after it. Use exactly these keys:

{{
  "title": "short document title",
  "risk_level": "Low" or "Medium" or "High",
  "summary": "plain-language explanation for a layperson",
  "applicable_laws": [{{"law": "Section/Act name", "explanation": "why it applies"}}],
  "deadlines": [{{"date_or_period": "...", "action": "...", "urgency": "High" or "Medium" or "Low"}}],
  "action_steps": ["3 to 5 clear next steps"]
}}

Rules:
- risk_level is how risky the document is for the person who received or
  signed it (High = urgent legal consequences or heavily one-sided terms).
- Only list deadlines that are actually in the document or set by law. If there
  are none, use an empty list [].
- Only cite laws you are confident apply. Do not invent section numbers.
- If the content is unreadable or not a legal document, still return the JSON:
  explain that in "summary", use risk_level "Low" and give sensible next steps.
- Write all values in simple English unless the document is clearly in Hindi,
  in which case write the values in Hindi. The keys must stay in English.
- You provide legal awareness, not formal legal advice.
"""

# ---------------------------------------------------------------
# 5. SET UP GEMINI (two "assistants" with different instructions)
# ---------------------------------------------------------------
genai.configure(api_key=GEMINI_API_KEY)

chat_model = genai.GenerativeModel(
    model_name=GEMINI_MODEL,
    system_instruction=CHAT_SYSTEM_INSTRUCTION,
)

analyze_model = genai.GenerativeModel(
    model_name=GEMINI_MODEL,
    system_instruction=ANALYZE_SYSTEM_INSTRUCTION,
    # Ask Gemini to return JSON; low temperature keeps answers consistent.
    generation_config={
        "response_mime_type": "application/json",
        "temperature": 0.2,
    },
)

# ---------------------------------------------------------------
# 6. CREATE THE FASTAPI APP + CORS
#    CORS lets browsers on any address talk to this server.
#    (Credentials stay off because "*" and credentials cannot be mixed.)
# ---------------------------------------------------------------
app = FastAPI(
    title="jatinSARTHI AI API",
    description="AI legal helper for Indian citizens",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------
# 7. DATA SHAPES FOR /chat
# ---------------------------------------------------------------
class ChatRequest(BaseModel):
    # The user's legal question (1 to 2000 characters)
    question: str = Field(..., min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    # The AI's plain-language answer
    answer: str


# ---------------------------------------------------------------
# 8. HELPER FUNCTIONS
# ---------------------------------------------------------------
def handle_gemini_error(error: Exception) -> HTTPException:
    """Turn a Gemini/network error into a friendly message for the user."""
    logger.error("Gemini request failed: %r", error)

    if isinstance(error, google_errors.ResourceExhausted):
        return HTTPException(
            status_code=429,
            detail="Too many requests right now. Please wait a minute and try again.",
        )
    if isinstance(error, (google_errors.Unauthenticated, google_errors.PermissionDenied)):
        return HTTPException(
            status_code=500,
            detail="The server's Gemini API key is not working. Please contact the site owner.",
        )
    if isinstance(error, google_errors.InvalidArgument):
        return HTTPException(
            status_code=400,
            detail="Gemini could not read this input. Try a clearer file or shorter text.",
        )
    return HTTPException(
        status_code=502,
        detail="Sorry, the AI could not respond right now. Please try again in a moment.",
    )


def get_response_text(response) -> str:
    """Safely read Gemini's reply (it can be empty if a safety filter blocks it)."""
    try:
        text = (response.text or "").strip()
    except ValueError:
        text = ""
    if not text:
        raise HTTPException(
            status_code=422,
            detail="The AI could not produce a reply for this input. Please rephrase or try a different file.",
        )
    return text


def strip_json_fences(text: str) -> str:
    """Remove ```json ... ``` wrappers and any text around the JSON object."""
    text = text.strip()
    # Remove opening ``` or ```json and closing ```
    text = re.sub(r"^```(?:json|JSON)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    # If there is still extra text, keep only from the first { to the last }
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        text = text[start:end + 1]
    return text


def pick_level(value, allowed, default):
    """Make sure a level is exactly one of the allowed words (e.g. 'High')."""
    cleaned = str(value or "").strip().capitalize()
    return cleaned if cleaned in allowed else default


def clean_analysis(data: dict) -> dict:
    """Make sure the result has exactly the keys and shapes the frontend expects."""
    laws = []
    for item in data.get("applicable_laws") or []:
        if isinstance(item, dict):
            laws.append({
                "law": str(item.get("law", "")).strip(),
                "explanation": str(item.get("explanation", "")).strip(),
            })
        elif isinstance(item, str):
            laws.append({"law": item.strip(), "explanation": ""})

    deadlines = []
    for item in data.get("deadlines") or []:
        if isinstance(item, dict):
            deadlines.append({
                "date_or_period": str(item.get("date_or_period", "")).strip(),
                "action": str(item.get("action", "")).strip(),
                "urgency": pick_level(item.get("urgency"), ("High", "Medium", "Low"), "Medium"),
            })

    steps = [str(s).strip() for s in (data.get("action_steps") or []) if str(s).strip()]

    return {
        "title": str(data.get("title") or "Untitled document").strip(),
        "risk_level": pick_level(data.get("risk_level"), ("Low", "Medium", "High"), "Medium"),
        "summary": str(data.get("summary") or "").strip(),
        "applicable_laws": laws,
        "deadlines": deadlines,
        "action_steps": steps[:5],
    }


# ---------------------------------------------------------------
# 9. ROUTE: GET /  - serve the website
# ---------------------------------------------------------------
@app.get("/", include_in_schema=False)
def home():
    if not INDEX_FILE.is_file():
        raise HTTPException(
            status_code=404,
            detail="index.html not found. Put it in the same folder as main.py.",
        )
    return FileResponse(INDEX_FILE, media_type="text/html")


# ---------------------------------------------------------------
# 10. ROUTE: GET /health  - quick check that the server is running
# ---------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "message": "jatinSARTHI AI server is running."}


# ---------------------------------------------------------------
# 11. ROUTE: POST /chat  - ask a legal question
# ---------------------------------------------------------------
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Please type a question.")

    try:
        response = await chat_model.generate_content_async(question)
        return ChatResponse(answer=get_response_text(response))
    except HTTPException:
        raise
    except Exception as error:
        raise handle_gemini_error(error)


# ---------------------------------------------------------------
# 12. ROUTE: POST /analyze  - analyse a document (file OR pasted text)
#     Send as "multipart/form-data" with a "file" and/or a "text" field.
# ---------------------------------------------------------------
@app.post("/analyze")
async def analyze(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
):
    text = (text or "").strip()
    has_file = file is not None and bool(file.filename)

    # We need at least a file or some text
    if not has_file and not text:
        raise HTTPException(
            status_code=400,
            detail="Please upload a file (PDF, PNG, JPG, JPEG, TXT) or paste some text.",
        )
    if len(text) > MAX_TEXT_CHARS:
        raise HTTPException(
            status_code=413,
            detail=f"Text is too long. Please keep it under {MAX_TEXT_CHARS:,} characters.",
        )

    # The pieces we will send to Gemini: an instruction plus the document
    parts = ["Analyse the following document and return the JSON report."]

    if has_file:
        extension = Path(file.filename).suffix.lower()
        if extension not in ALLOWED_FILE_TYPES:
            raise HTTPException(
                status_code=415,
                detail="Unsupported file type. Please upload a PDF, PNG, JPG, JPEG or TXT file.",
            )

        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")
        if len(data) > MAX_FILE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File is too large. Maximum size is {MAX_FILE_BYTES // (1024 * 1024)} MB.",
            )

        if extension == ".txt":
            # Text files are sent as plain text
            try:
                file_text = data.decode("utf-8")
            except UnicodeDecodeError:
                file_text = data.decode("latin-1", errors="replace")
            if len(file_text) > MAX_TEXT_CHARS:
                raise HTTPException(
                    status_code=413,
                    detail=f"Text file is too long. Please keep it under {MAX_TEXT_CHARS:,} characters.",
                )
            parts.append(f"--- Document text ---\n{file_text}")
        else:
            # PDFs and images are sent as raw bytes
            parts.append({"mime_type": ALLOWED_FILE_TYPES[extension], "data": data})

    if text:
        parts.append(f"--- Additional text provided by the user ---\n{text}")

    # Ask Gemini, then turn its reply into clean JSON
    try:
        response = await analyze_model.generate_content_async(parts)
        raw_reply = get_response_text(response)
    except HTTPException:
        raise
    except Exception as error:
        raise handle_gemini_error(error)

    try:
        parsed = json.loads(strip_json_fences(raw_reply))
        if not isinstance(parsed, dict):
            raise ValueError("JSON was not an object")
    except ValueError:
        logger.error("Gemini returned invalid JSON: %s", raw_reply[:500])
        raise HTTPException(
            status_code=502,
            detail="The AI returned an unreadable report. Please try again.",
        )

    return clean_analysis(parsed)