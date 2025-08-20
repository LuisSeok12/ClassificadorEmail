
import io
import os
import re
from typing import Optional
from fastapi import UploadFile, HTTPException
import pdfplumber

ALLOWED_EXT = {".txt", ".pdf"}

def safe_truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars - 3] + "..."

async def read_email_payload(file: Optional[UploadFile], text: Optional[str]) -> str:
    """
    Lê e normaliza o conteúdo enviado (arquivo .txt/.pdf ou textarea).
    """
    if text and text.strip():
        return text.strip()

    if file:
        name = file.filename or "upload"
        ext = os.path.splitext(name)[1].lower()
        if ext not in ALLOWED_EXT:
            raise HTTPException(status_code=400, detail=f"Extensão não suportada: {ext}. Use .txt ou .pdf")
        content = await file.read()
        if ext == ".txt":
            try:
                return content.decode("utf-8", errors="ignore")
            except Exception:
                return content.decode("latin-1", errors="ignore")
        elif ext == ".pdf":
            try:
                with pdfplumber.open(io.BytesIO(content)) as pdf:
                    pages = [p.extract_text() or "" for p in pdf.pages]
                return "\n".join(pages)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Falha ao ler PDF: {e}")
    return ""
