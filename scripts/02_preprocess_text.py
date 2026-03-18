"""
02_preprocess_text.py
---------------------
Pré-processamento de texto para análise de sentimentos.

Lê o arquivo cleaned_comments.csv (gerado pelo 01_clean_data.py) e adiciona
duas novas colunas de texto tratado:

    text_clean        — limpeza leve, para uso com BERTimbau.
    text_preprocessed — limpeza completa, para modelos tradicionais (TF-IDF).

Por que duas versões?
    BERTimbau é um modelo Transformer treinado em português real, com emojis,
    gírias e estrutura contextual. Remover stopwords e lematizar o texto antes
    de passá-lo ao BERTimbau piora o desempenho, pois destrói o contexto que
    o modelo aprendeu. Por isso, `text_clean` faz apenas a remoção de ruídos
    técnicos (URLs, menções, HTML).

    Modelos tradicionais como TF-IDF + Regressão Logística dependem de
    frequência de palavras, então stopwords e formas flexionadas criam
    esparsidade desnecessária. Para eles, `text_preprocessed` aplica
    normalização completa incluindo stemming com RSLP (português).

Entrada:
    data/processed/cleaned_comments.csv

Saída:
    data/processed/preprocessed_comments.csv

Dependências extras necessárias:
    pip install nltk emoji

Uso:
    python scripts/02_preprocess_text.py
"""

import os
import re
import html
import unicodedata

import pandas as pd
import nltk
import emoji as emoji_lib

# ─── Download automático de recursos NLTK ────────────────────────────────────
# Executado apenas se ainda não estiverem presentes.
for resource, path in [
    ("stopwords",          "corpora/stopwords"),
    ("punkt",              "tokenizers/punkt"),
    ("punkt_tab",          "tokenizers/punkt_tab"),
    ("rslp",               "stemmers/rslp"),
]:
    try:
        nltk.data.find(path)
    except LookupError:
        print(f"  Baixando recurso NLTK: {resource} ...")
        nltk.download(resource, quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import RSLPStemmer

STOPWORDS_PT = set(stopwords.words("portuguese"))
STEMMER      = RSLPStemmer()


# ─── Caminhos ────────────────────────────────────────────────────────────────

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_CSV  = os.path.join(ROOT, "data", "processed", "cleaned_comments.csv")
OUTPUT_CSV = os.path.join(ROOT, "data", "processed", "preprocessed_comments.csv")


# ─── Funções de limpeza ───────────────────────────────────────────────────────

# Padrão para URL (http/https e www sem protocolo)
_RE_URL       = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
# Menções @usuario
_RE_MENTION   = re.compile(r"@\w+")
# Hashtags #palavra
_RE_HASHTAG   = re.compile(r"#\w+")
# Entidades HTML (&amp; &lt; etc.)
_RE_HTML_ENT  = re.compile(r"&[a-zA-Z]+;")
# Espaços múltiplos / quebras de linha desnecessárias
_RE_SPACES    = re.compile(r"\s+")
# Caracteres não-linguísticos para text_preprocessed
# Mantemos letras latinas (incluindo acentos), números e espaço.
_RE_NOISE     = re.compile(r"[^a-zA-ZÀ-ÖØ-öø-ÿ0-9\s]")


def clean_text(text: str) -> str:
    """
    Limpeza leve — adequada para BERTimbau.

    Remove:
        - Entidades HTML (&amp; → & → depois tratadas)
        - URLs
        - Menções (@usuario)
        - Hashtags (#tópico)
        - Emojis (convertidos para descrição textual em português, ex: 💛 → 'coração amarelo')
        - Espaços e quebras de linha extras

    Preserva:
        - Maiúsculas/minúsculas (BERTimbau é case-sensitive)
        - Pontuação (dá contexto ao Transformer)
        - Acentos e caracteres especiais do português
    """
    if not isinstance(text, str):
        return ""

    # Decodifica entidades HTML
    text = html.unescape(text)

    # Remove URLs
    text = _RE_URL.sub(" ", text)

    # Remove menções
    text = _RE_MENTION.sub(" ", text)

    # Remove hashtags
    text = _RE_HASHTAG.sub(" ", text)

    # Converte emojis em descrição textual
    # Ex: 💛 → ':coração_amarelo:' — preserva a carga semântica/sentimental
    text = emoji_lib.demojize(text, language="pt")

    # Normaliza espaços
    text = _RE_SPACES.sub(" ", text).strip()

    return text


def preprocess_text(text: str) -> str:
    """
    Pré-processamento completo — adequado para modelos tradicionais (TF-IDF).

    Etapas adicionais em relação a clean_text():
        1. Conversão para minúsculas
        2. Remoção de caracteres não-linguísticos (pontuação, números, símbolos)
        3. Tokenização
        4. Remoção de stopwords em português
        5. Stemming com RSLP (Removedor de Sufixos da Língua Portuguesa)

    Nota sobre stemming vs. lematização:
        A lematização retorna a forma canônica da palavra (ex: "correndo" → "correr"),
        enquanto o stemming remove sufixos de forma mais agressiva (ex: "correndo" → "corr").
        O RSLP é o stemmer padrão para português no NLTK. Para lematização real,
        seria necessário o modelo spaCy 'pt_core_news_sm', o que adicionaria
        uma dependência pesada. Para os fins deste TCC, o RSLP é suficiente como
        baseline.
    """
    if not isinstance(text, str):
        return ""

    # Parte da limpeza leve
    text = html.unescape(text)
    text = _RE_URL.sub(" ", text)
    text = _RE_MENTION.sub(" ", text)
    text = _RE_HASHTAG.sub(" ", text)

    # Remove emojis completamente (não convertemos para texto aqui)
    text = emoji_lib.replace_emoji(text, replace=" ")

    # Minúsculas
    text = text.lower()

    # Remove acentos → normalização NFKD + filtragem de caracteres não-ASCII
    # Isso simplifica a comparação lexical em TF-IDF
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))

    # Remove tudo que não é letra ou espaço
    text = _RE_NOISE.sub(" ", text)

    # Tokeniza
    try:
        tokens = word_tokenize(text, language="portuguese")
    except Exception:
        tokens = text.split()

    # Remove stopwords e tokens curtos (1-2 caracteres)
    tokens = [t for t in tokens if t not in STOPWORDS_PT and len(t) > 2]

    # Stemming
    tokens = [STEMMER.stem(t) for t in tokens]

    return " ".join(tokens)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 50)
    print("  Pre-processamento de texto - TCC")
    print("=" * 50)

    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(
            f"Arquivo não encontrado: {INPUT_CSV}\n"
            "Execute primeiro: python scripts/01_clean_data.py"
        )

    df = pd.read_csv(INPUT_CSV)
    total = len(df)
    print(f"\n  Carregados {total} comentários de '{INPUT_CSV}'")

    # Aplica limpeza leve (para BERTimbau)
    print("  Aplicando clean_text (limpeza leve para BERTimbau)...")
    df["text_clean"] = df["text"].apply(clean_text)

    # Aplica pré-processamento completo (para TF-IDF)
    print("  Aplicando preprocess_text (pré-processamento completo para TF-IDF)...")
    df["text_preprocessed"] = df["text"].apply(preprocess_text)

    # Estatísticas
    empty_clean   = (df["text_clean"].str.strip() == "").sum()
    empty_preproc = (df["text_preprocessed"].str.strip() == "").sum()
    print(f"\n  text_clean vazio após limpeza     : {empty_clean}")
    print(f"  text_preprocessed vazio após NLP  : {empty_preproc}")
    print(f"  (Comentários apenas-emoji perdem conteúdo em text_preprocessed)")

    # Salva
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n  Salvo: {OUTPUT_CSV}")
    print("\n  Pré-processamento concluído.")


if __name__ == "__main__":
    main()
