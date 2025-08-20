
import os
import json
import re
from typing import Dict
import requests
from dotenv import load_dotenv

load_dotenv()

# ---------- Config ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN", "")
HF_MODEL = os.getenv("HF_ZERO_SHOT_MODEL", "joeddav/xlm-roberta-large-xnli")  

CANDIDATES = ["Produtivo", "Improdutivo"]

# ---------- Helpers ----------
def _heuristics_category(text: str) -> Dict:
    """
    Classificador fallback baseado em palavras-chave.
    """
    t = " " + text.lower() + " "
    productive_keys = [
        "suporte", "erro", "bug", "falha", "problema", "nao consigo",
        "acesso", "senha", "login", "urgente", "prazo", "protocolo", "ticket",
        "atualizacao", "andamento", "status", "orcamento", "financeiro",
        "anexo", "documento", "fatura", "nota fiscal", "nf", "proposta"
    ]
    noise_keys = [
        "obrigado", "agradeco", "bom dia", "boa tarde", "boa noite",
        "feliz", "parabens", "natal", "ano novo", "att", "atenciosamente"
    ]

    score = 0
    for k in productive_keys:
        if f" {k} " in t:
            score += 1
    for k in noise_keys:
        if f" {k} " in t:
            score -= 0.3

    category = "Produtivo" if score >= 0.5 else "Improdutivo"
    confidence = min(0.95, max(0.55, 0.55 + (score/10.0)))
    return {"category": category, "confidence": float(round(confidence, 3)), "classified_by": "heuristics"}

async def _openai_classify(text: str) -> Dict:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY ausente.")
    url = f"{OPENAI_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    prompt = f"""
Você é um classificador. Responda APENAS em JSON.
Categorias possíveis: {CANDIDATES}.
Classifique o e-mail abaixo como "Produtivo" ou "Improdutivo" e forneça um confidence entre 0 e 1.

Email:
\"\"\"
{text[:4000]}
\"\"\"

Formato de saída (JSON): {{"category": "Produtivo|Improdutivo", "confidence": 0.0}}
"""
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": "Você classifica e-mails de maneira confiável e concisa."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0
    }
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    content = data["choices"][0]["message"]["content"]
    # tentar parsear JSON
    try:
        parsed = json.loads(content)
        category = parsed.get("category", "Produtivo")
        confidence = float(parsed.get("confidence", 0.75))
    except Exception:
        # fallback leve
        category = "Produtivo" if "anexo" in text.lower() or "suporte" in text.lower() else "Improdutivo"
        confidence = 0.7
    return {"category": category, "confidence": confidence, "classified_by": "openai"}

async def _hf_zeroshot_classify(text: str) -> Dict:
    if not HF_TOKEN:
        raise RuntimeError("HUGGINGFACE_API_TOKEN ausente.")
    url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}
    payload = {
        "inputs": text[:4000],
        "parameters": {
            "candidate_labels": CANDIDATES,
            "hypothesis_template": "Este texto é {}."
        }
    }
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    try:
        labels = data["labels"]
        scores = data["scores"]
        best_idx = int(scores.index(max(scores)))
        category = labels[best_idx]
        confidence = float(scores[best_idx])
        return {"category": category, "confidence": confidence, "classified_by": "huggingface-inference"}
    except Exception as e:
        # fallback heurístico
        return _heuristics_category(text)

async def classify_email(text: str) -> Dict:
    """
    Orquestração: tenta HF -> OpenAI -> heurística, dependendo das credenciais disponíveis.
    """
    # Preferência: HF (zero-shot multilíngue) se token presente
    if HF_TOKEN:
        try:
            return await _hf_zeroshot_classify(text)
        except Exception:
            pass
    # Em seguida OpenAI
    if OPENAI_API_KEY:
        try:
            return await _openai_classify(text)
        except Exception:
            pass
    # Fallback heurístico
    return _heuristics_category(text)

async def suggest_reply(category: str, original_text: str) -> Dict:
    """
    Gera uma resposta curta com OpenAI (se disponível) ou usa templates.
    """
    if OPENAI_API_KEY:
        try:
            url = f"{OPENAI_BASE_URL}/chat/completions"
            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
            system = "Você escreve respostas de e-mail curtas, claras, empáticas e profissionais em PT-BR."
            user = f"""
Contexto: O e-mail foi classificado como {category}.
Escreva uma resposta de até 6 linhas, objetiva. Se for Produtivo, solicite informações mínimas necessárias,
cite anexos se houver menção e proponha próximo passo. Se for Improdutivo, responda cordialmente e encerre.
E-mail original (resuma e responda):

\"\"\"
{original_text[:4000]}
\"\"\"
"""
            payload = {"model": OPENAI_MODEL, "messages": [{"role":"system","content":system},{"role":"user","content":user}], "temperature": 0.3}
            r = requests.post(url, headers=headers, json=payload, timeout=60)
            r.raise_for_status()
            data = r.json()
            content = data["choices"][0]["message"]["content"].strip()
            return {"text": content, "provider": "openai"}
        except Exception:
            pass

    # Templates
    if category.lower().startswith("produt"):
        text = (
            "Olá! Obrigado pelo contato. Para dar sequência, poderia nos enviar o número do protocolo "
            "ou mais detalhes (prints/arquivos) sobre o ocorrido? Assim que recebermos essas informações, "
            "abriremos/atualizaremos o ticket e retornaremos com o status e próximos passos. "
            "Ficamos à disposição."
        )
    else:
        text = (
            "Olá! Agradecemos a mensagem. Registramos seu contato. Caso precise de ajuda ou tenha alguma "
            "solicitação específica, fale conosco por este canal. Tenha um ótimo dia!"
        )
    return {"text": text, "provider": "template"}
