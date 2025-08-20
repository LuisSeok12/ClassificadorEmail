# AutoU — Classificador de E-mails (Demo)

Aplicação **FastAPI** com interface HTML (Tailwind via CDN) para **classificar e-mails** em _Produtivo_ ou _Improdutivo_ e **sugerir resposta automática**.

- **Classificação**: usa **Hugging Face Inference API** (zero-shot multilíngue) se disponível, senão **OpenAI**, e por último um **fallback heurístico**.
- **Resposta**: usa **OpenAI** se disponível; caso contrário, templates prontos.

## ⚙️ Executar localmente

```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Opcional: copie .env.example para .env e configure credenciais
# setx OPENAI_API_KEY "sua_chave"   # Windows (sessão futura)
# export OPENAI_API_KEY="sua_chave"  # Linux/Mac

uvicorn app.main:app --reload --port 7860
# Abra http://localhost:7860
```

## 📁 Estrutura

```
app/
 ├── main.py          # Endpoints FastAPI
 ├── nlp.py           # Pré-processamento simples (stopwords, acentuação, etc.)
 ├── providers.py     # Integração com HF / OpenAI + heurísticas
 └── utils.py         # Leitura de .txt/.pdf e utilidades
templates/
 └── index.html       # UI (Tailwind CDN + fetch API)
static/
 └── style.css
sample_emails/
 ├── suporte_status.txt
 └── feliz_natal.txt
requirements.txt
Procfile
runtime.txt
.env.example
README.md
```

## Deploy rápido (Render.com)

1. **Crie um repositório no GitHub** com estes arquivos.
2. No **Render**, crie um **Web Service** a partir do repositório.
3. Defina:
   - **Runtime**: Python 3.11
   - **Start command**: `gunicorn -k uvicorn.workers.UvicornWorker app.main:app`
   - **Environment variables**: defina `OPENAI_API_KEY` e/ou `HUGGINGFACE_API_TOKEN`.
4. Abra a URL gerada (ex.: `https://seu-app.onrender.com`).

> Alternativas: Railway.app, Deta.space, Fly.io, Hugging Face Spaces (pede refatoração para Gradio/Streamlit), ou Heroku.

## Como funciona (resumo técnico)

1. **Pré-processamento** (`app/nlp.py`): normalização (lower/acentos), remoção de stopwords PT-BR, contagem de tokens.
2. **Classificação** (`app/providers.py`):
   - Tenta **Hugging Face Inference** (`joeddav/xlm-roberta-large-xnli`).
   - Se indisponível, usa **OpenAI** (`gpt-4o-mini` por padrão).
   - Fallback: **heurísticas** por palavras-chave.
3. **Geração de resposta**:
   - **OpenAI** (quando disponível) com prompt objetivo.
   - Fallback: **templates** prontos.

## Extensões sugeridas

- Persistir resultados (SQLite / Postgres).
- Painel de métricas (acurácia, distribuição por categoria).
- Treinar um classificador leve supervisionado com dados reais (LogReg/SVM) como camada base.
- Filas (Celery/RQ) para processamento em lote.

## Observações

- Não suba a sua chave de API no GitHub.
- Para PDFs digitalizados (scans), adicione OCR (Tesseract/pytesseract).

---

**Autor:** Luis Eduardo Fonseca de Almeida Sarinho — Projeto de demonstração para o processo seletivo AutoU.
