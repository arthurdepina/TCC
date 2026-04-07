"""
01_clean_data.py
----------------
Limpeza estrutural dos dados brutos coletados via YouTube Data API.

Este script trata problemas de qualidade que existem independentemente de NLP:
nulos, duplicatas, tipos de dados incorretos e inconsistências de schema.

Entradas:
    raw_comments.csv       — comentários brutos
    video_metadata.csv     — metadados dos vídeos

Saídas (em data/processed/):
    cleaned_comments.csv   — comentários limpos
    cleaned_metadata.csv   — metadados limpos

Uso:
    python scripts/01_clean_data.py
"""

import os
import re
import pandas as pd


# ─── Caminhos ────────────────────────────────────────────────────────────────

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_COMMENTS  = os.path.join(ROOT, "raw_comments.csv")
RAW_METADATA  = os.path.join(ROOT, "video_metadata.csv")
OUTPUT_DIR    = os.path.join(ROOT, "data", "processed")


# ─── Limpeza de comentários ───────────────────────────────────────────────────

def clean_comments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica limpeza estrutural ao DataFrame de comentários.

    Etapas:
        1. Remove linhas com texto nulo.
        2. Remove comentários cujo texto é vazio após strip.
        3. Remove comentIds duplicados (mantém a primeira ocorrência).
        4. Converte publishedAt para datetime com fuso UTC.
        5. Garante que likeCount seja inteiro não-negativo.
        6. Cria coluna auxiliar `text_length` (nº de caracteres do texto original).
        7. Cria coluna auxiliar `is_substantive` — True quando o comentário contém
           ao menos uma palavra (sequência de letras), não apenas emojis/pontuação.
    """
    report = {}

    # ── 1. Nulos ──────────────────────────────────────────────────────────────
    n_null = df["text"].isna().sum()
    df = df.dropna(subset=["text"]).copy()
    report["nulos removidos"] = int(n_null)

    # ── 2. Textos vazios após strip ───────────────────────────────────────────
    df["text"] = df["text"].str.strip()
    n_empty = (df["text"] == "").sum()
    df = df[df["text"] != ""]
    report["vazios após strip"] = int(n_empty)

    # ── 3. Duplicatas por commentId ───────────────────────────────────────────
    n_dup = df.duplicated(subset=["commentId"]).sum()
    df = df.drop_duplicates(subset=["commentId"])
    report["commentIds duplicados removidos"] = int(n_dup)

    # ── 4. Tipo de data ───────────────────────────────────────────────────────
    df["publishedAt"] = pd.to_datetime(df["publishedAt"], utc=True)

    # ── 5. likeCount não-negativo ─────────────────────────────────────────────
    n_neg = (df["likeCount"] < 0).sum()
    df = df[df["likeCount"] >= 0]
    report["likeCount negativo removidos"] = int(n_neg)

    # ── 6. Coluna auxiliar: comprimento do texto ──────────────────────────────
    df["text_length"] = df["text"].str.len()

    # ── 7. Flag: comentário com ao menos uma palavra (letra) ──────────────────
    # Comentários como "💛💛💛" são válidos e portadores de sentimento,
    # mas não contêm palavras textuais. Esta flag permite filtrá-los nas
    # etapas seguintes sem descartá-los permanentemente aqui.
    df["is_substantive"] = df["text"].str.contains(r"[a-zA-ZÀ-ÖØ-öø-ÿ]", regex=True)

    df = df.reset_index(drop=True)

    _print_report("Comentários", report, original=len(df) + sum(report.values()), final=len(df))
    return df


# ─── Limpeza de metadados ─────────────────────────────────────────────────────

def clean_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica limpeza estrutural ao DataFrame de metadados de vídeos.

    Etapas:
        1. Preenche descrições nulas com string vazia.
        2. Remove espaços extras em colunas de texto.
        3. Remove vídeos duplicados por videoId.
        4. Converte publishedAt para datetime com fuso UTC.
        5. Garante que contadores (viewCount, likeCount, commentCount) sejam
           inteiros não-negativos.
        6. Adiciona coluna `channel_type` inferida a partir do channelTitle,
           usando o mesmo mapeamento da coleta original.
    """
    report = {}

    # ── 1. Descrições nulas ───────────────────────────────────────────────────
    n_null_desc = df["description"].isna().sum()
    df["description"] = df["description"].fillna("")
    report["descrições nulas preenchidas"] = int(n_null_desc)

    # ── 2. Strip em colunas de texto ──────────────────────────────────────────
    for col in ["title", "description", "channelTitle"]:
        df[col] = df[col].str.strip()

    # ── 3. Duplicatas por videoId ─────────────────────────────────────────────
    n_dup = df.duplicated(subset=["videoId"]).sum()
    df = df.drop_duplicates(subset=["videoId"])
    report["videoIds duplicados removidos"] = int(n_dup)

    # ── 4. Tipo de data ───────────────────────────────────────────────────────
    df["publishedAt"] = pd.to_datetime(df["publishedAt"], utc=True)

    # ── 5. Contadores não-negativos ───────────────────────────────────────────
    for col in ["viewCount", "likeCount", "commentCount"]:
        n_neg = (df[col] < 0).sum()
        if n_neg:
            df = df[df[col] >= 0]
            report[f"{col} negativo removidos"] = int(n_neg)

    # ── 6. Coluna channel_type ────────────────────────────────────────────────
    # Mapeamento baseado nos canais definidos na proposta do TCC.
    profissional_channels = {
        "Prazer, Karnal - Canal Oficial de Leandro Karnal",
        "Maria Homem",
        "Neurologia e Psiquiatria",
        "Rossandro Klinjey",
        "PodPeople - Ana Beatriz Barbosa",
        "Augusto Cury",
        "Minutos Psíquicos",
        "Casa do Saber",
    }
    df["channel_type"] = df["channelTitle"].apply(
        lambda ch: "profissional" if ch in profissional_channels else "amador"
    )

    df = df.reset_index(drop=True)

    _print_report("Metadados", report, original=len(df), final=len(df))
    return df


# ─── Utilitários ──────────────────────────────────────────────────────────────

def _print_report(name: str, report: dict, original: int, final: int) -> None:
    print(f"\n{'-' * 50}")
    print(f"  {name}")
    print(f"{'-' * 50}")
    print(f"  Linhas originais : {original:>6}")
    for k, v in report.items():
        if v:
            print(f"  {k:<35}: {v:>6}")
    print(f"  Linhas finais    : {final:>6}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 50)
    print("  Limpeza estrutural de dados - TCC")
    print("=" * 50)

    # Comentários
    comments = pd.read_csv(RAW_COMMENTS)
    comments = clean_comments(comments)
    out = os.path.join(OUTPUT_DIR, "cleaned_comments.csv")
    comments.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n  Salvo: {out}")

    # Metadados
    metadata = pd.read_csv(RAW_METADATA)
    metadata = clean_metadata(metadata)
    out = os.path.join(OUTPUT_DIR, "cleaned_metadata.csv")
    metadata.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"  Salvo: {out}")

    print("\n  Limpeza concluída.")


if __name__ == "__main__":
    main()
