
import re
import unicodedata
from unidecode import unidecode

PT_STOPWORDS = set("""a à após ao aos aoonde aquela aquelas aquele aqueles aquilo as até com como contra da das de dela delas dele deles depois desde do dos e é em enquanto entre era eram essa essas esse esses esta estava estavam estão eu faz fazem fizeram foi foram fora haja há isso isto já la lá lhe lhes mais mas me mesmo meu meus minha minhas na nas não nem no nos nós o os ou para pela pelas pelo pelos por porque porém posso primeira primeiro quais qual quando que quem se sem sempre sendo seu seus sob sobre sua suas também tão te tem têm tendo então tua tuas um uma umas uns você vocês""".split())

def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def strip_accents_lower(text: str) -> str:

    return unidecode(text.lower())

def remove_stopwords(text: str) -> str:
    tokens = re.findall(r"[a-zA-ZÀ-ÿ0-9@._-]+", text, flags=re.UNICODE)
    filtered = [t for t in tokens if t.lower() not in PT_STOPWORDS]
    return " ".join(filtered)

def preprocess_text(text: str) -> dict:
    """
    Pipeline simples: lower, remover acentos, normalizar espaços, remover stopwords.
    Retorna dict com texto limpo e contagem tosca de "tokens" (palavras).
    """
    if not text:
        text = ""
    text_norm = normalize_whitespace(text)
    text_clean = strip_accents_lower(text_norm)
    text_nostop = remove_stopwords(text_clean)
    num_tokens = len(text_nostop.split())
    return {"clean_text": text_nostop, "num_tokens": num_tokens}
