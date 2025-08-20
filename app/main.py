
import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from app.utils import read_email_payload, safe_truncate
from app.nlp import preprocess_text
from app.providers import classify_email, suggest_reply

app = FastAPI(title="AutoU - Classificador de E-mails", version="1.0.0")

app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def index():

    return FileResponse("templates/index.html")

class AnalyzeResponse(BaseModel):
    category: str
    confidence: float
    suggested_reply: str
    classified_by: str
    tokens: int
    preview: str

@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(
    file: Optional[UploadFile] = File(default=None),
    text: Optional[str] = Form(default=None)
):
    if not file and not text:
        raise HTTPException(status_code=400, detail="Envie um arquivo (.txt/.pdf) ou cole o texto do e-mail.")

    # 1) Ler conteúdo
    raw_text = await read_email_payload(file, text)

    if not raw_text or raw_text.strip() == "":
        raise HTTPException(status_code=400, detail="Conteúdo vazio após leitura do arquivo/textarea.")

    # 2) Pré-processar (NLP simples)
    pre = preprocess_text(raw_text)

    # 3) Classificar (API de IA se disponível; senão heurística)
    cls = await classify_email(pre["clean_text"])

    # 4) Sugerir resposta (usa API de IA se disponível; senão template)
    reply = await suggest_reply(cls["category"], raw_text)

    # 5) Construir resposta
    return JSONResponse({
        "category": cls["category"],
        "confidence": round(cls["confidence"], 4),
        "suggested_reply": reply["text"],
        "classified_by": cls["classified_by"],
        "tokens": pre["num_tokens"],
        "preview": safe_truncate(raw_text, 600)
    })
