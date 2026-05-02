"""
07_build_final_dataset.py
--------------------------
Monta o dataset final para análise, cruzando as predições da 2ª metade
com os metadados dos comentários e calculando palavras-chave de confiabilidade.

A base de análise é composta exclusivamente pelos comentários da 2ª metade —
aqueles que o XGBoost nunca viu durante o treinamento (XGBoost_half.pkl foi
treinado apenas na 1ª metade). Isso garante que os rótulos XGBoost são
predições genuínas, não memorização.

Entradas:
    data/processed/second_half_predictions.csv
    data/processed/preprocessed_comments.csv

Saídas:
    data/processed/final_dataset.csv

Colunas do dataset final:
    commentId    — ID único do comentário
    text         — texto original
    text_clean   — texto tratado (URLs removidas, emojis convertidos)
    channel_type — "profissional" ou "amador"
    likeCount    — número de curtidas
    video_title  — título do vídeo
    label_llm    — rótulo atribuído pelo LLM (referência)
    label_xgb    — rótulo predito pelo XGBoost (predição genuína)
    confidence   — probabilidade máxima do XGBoost (0,0–1,0)
    intensidade  — ALTA (>0,80) | MEDIA (0,60–0,80) | BAIXA (<0,60)
    concorda     — True se label_llm == label_xgb
    keywords     — termos de confiabilidade percebida encontrados em text_clean
                   (separados por ";"; vazio se nenhum encontrado)

Uso:
    python scripts/07_build_final_dataset.py

Requisitos:
    pip install pandas
"""

import sys
import pandas as pd
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ─── Configurações ────────────────────────────────────────────────────────────

TERMOS_CONFIABILIDADE = [
    "explicou", "explicação", "faz sentido", "confio", "confiança",
    "científico", "embasado", "comprovado", "especialista", "referência",
    "pesquisa", "estudo", "dados", "evidência", "profissional",
    "recomendo", "recomenda", "seguro", "correto", "verdade",
]

# ─── Caminhos ────────────────────────────────────────────────────────────────

ROOT         = Path(__file__).parent.parent
PREDS_CSV    = ROOT / "data" / "processed" / "second_half_predictions.csv"
COMMENTS_CSV = ROOT / "data" / "processed" / "preprocessed_comments.csv"
OUT_FINAL    = ROOT / "data" / "processed" / "final_dataset.csv"


# ─── Palavras-chave ───────────────────────────────────────────────────────────

def extrair_keywords(text: str) -> str:
    t = str(text).lower()
    encontrados = [term for term in TERMOS_CONFIABILIDADE if term in t]
    return ";".join(encontrados)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("Carregando dados...")
    preds_df    = pd.read_csv(PREDS_CSV)
    comments_df = pd.read_csv(COMMENTS_CSV, usecols=[
        "commentId", "text", "text_clean", "channel_type", "likeCount", "video_title"
    ])

    print(f"  Predições (2ª metade) : {len(preds_df):,} comentários")
    print(f"  Comentários (total)   : {len(comments_df):,}")

    # ── Join: predições + metadados ─────────────────────────────────────────
    final_df = preds_df.merge(comments_df, on="commentId", how="inner")
    print(f"  Após join             : {len(final_df):,} comentários")

    if len(final_df) != len(preds_df):
        print(f"  AVISO: {len(preds_df) - len(final_df)} predições sem metadados correspondentes.")

    # ── Palavras-chave de confiabilidade ────────────────────────────────────
    print("\nCalculando palavras-chave de confiabilidade...")
    final_df["keywords"] = final_df["text_clean"].apply(extrair_keywords)

    n_com = (final_df["keywords"] != "").sum()
    print(f"  Com pelo menos 1 termo: {n_com:,} ({n_com/len(final_df)*100:.1f}%)")

    # ── Ordena colunas ──────────────────────────────────────────────────────
    final_df = final_df[[
        "commentId", "text", "text_clean", "channel_type", "likeCount", "video_title",
        "label_llm", "label_xgb", "confidence", "intensidade", "concorda", "keywords",
    ]]

    # ── Resumo ──────────────────────────────────────────────────────────────
    total = len(final_df)
    print(f"\n{'='*60}")
    print(f"  Dataset final: {total:,} comentários (2ª metade — avaliação genuína)")
    print(f"{'='*60}")

    print("\nDistribuição por label_xgb:")
    for lbl, cnt in final_df["label_xgb"].value_counts().items():
        print(f"  {lbl:<12} {cnt:>6}  ({cnt/total*100:.1f}%)")

    print("\nDistribuição por label_llm:")
    for lbl, cnt in final_df["label_llm"].value_counts().items():
        print(f"  {lbl:<12} {cnt:>6}  ({cnt/total*100:.1f}%)")

    print("\nPor channel_type:")
    for ct, cnt in final_df["channel_type"].value_counts().items():
        print(f"  {ct:<14} {cnt:>6}  ({cnt/total*100:.1f}%)")

    concorda_pct = final_df["concorda"].mean() * 100
    print(f"\nConcordância LLM × XGBoost: {concorda_pct:.1f}%")

    print(f"\nConfiança média por label_xgb:")
    for lbl, mean_c in final_df.groupby("label_xgb")["confidence"].mean().round(3).items():
        print(f"  {lbl:<12} {mean_c:.3f}")

    # ── Salva ───────────────────────────────────────────────────────────────
    final_df.to_csv(OUT_FINAL, index=False, encoding="utf-8-sig")
    print(f"\nDataset final salvo: {OUT_FINAL}")
    print("Próximo passo: python scripts/08_analyze_results.py")


if __name__ == "__main__":
    main()
