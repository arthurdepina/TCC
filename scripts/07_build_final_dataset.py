"""
07_build_final_dataset.py
--------------------------
Monta o dataset unificado final (Opcao A do TCC).

Sob a Opcao A, o LLM rotulou 100% dos 42.372 comentarios.
O XGBoost foi treinado via cross-validation sobre os 34.491 nao-DESCARTAVEL
e avaliado pelos resultados de CV (F1-macro 73.3% — ver relatorio_treino.txt).

Este script:
    1. Calcula confianca (predict_proba) para os 34.491 nao-DESCARTAVEL,
       usando os embeddings.npy ja gerados pelo script 05
    2. Reporta concordancia XGBoost x LLM (analise informativa)
    3. Constroi o dataset unificado final com labels LLM + confianca XGBoost

Entradas:
    data/processed/preprocessed_comments.csv
    data/processed/llm_labeled_comments_v3.csv
    data/processed/embeddings.npy
    data/processed/embedding_meta.csv
    data/processed/models/XGBoost.pkl

Saidas:
    data/processed/all_labeled_comments.csv

Colunas do dataset final:
    commentId, text, text_clean, channel_type, likeCount, video_title,
    label, confidence

    label      -- rotulo LLM (fonte principal para a analise)
    confidence -- probabilidade maxima do XGBoost (None para DESCARTAVEL)
                  usada para calcular intensidade emocional no script 08:
                    > 0.80 -> ALTA | 0.60-0.80 -> MEDIA | < 0.60 -> BAIXA

Uso:
    python scripts/07_build_final_dataset.py
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ─── Configurações ────────────────────────────────────────────────────────────

LABEL_MAP_REVERSE = {0: "NEGATIVO", 1: "TANGENCIAL", 2: "POSITIVO"}
LABEL_ORDER       = ["NEGATIVO", "TANGENCIAL", "POSITIVO"]

# ─── Caminhos ────────────────────────────────────────────────────────────────

ROOT         = Path(__file__).parent.parent
COMMENTS_CSV = ROOT / "data" / "processed" / "preprocessed_comments.csv"
LLM_CSV      = ROOT / "data" / "processed" / "llm_labeled_comments_v3.csv"
EMBEDDINGS   = ROOT / "data" / "processed" / "embeddings.npy"
META_CSV     = ROOT / "data" / "processed" / "embedding_meta.csv"
MODEL_PATH   = ROOT / "data" / "processed" / "models" / "XGBoost.pkl"
OUT_FINAL    = ROOT / "data" / "processed" / "all_labeled_comments.csv"


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    # ── Carrega dados ───────────────────────────────────────────────────────
    print("Carregando dados...")
    comments_df = pd.read_csv(COMMENTS_CSV, usecols=[
        "commentId", "text", "text_clean", "channel_type", "likeCount", "video_title"
    ])
    llm_df  = pd.read_csv(LLM_CSV).rename(columns={"label_llm": "label"})
    meta_df = pd.read_csv(META_CSV)

    print(f"  Comentarios totais    : {len(comments_df):,}")
    print(f"  Rotulados pelo LLM    : {len(llm_df):,}")
    print(f"  Com embeddings (script 05): {len(meta_df):,}  (exclui DESCARTAVEL)")

    # ── Carrega XGBoost ─────────────────────────────────────────────────────
    print("\nCarregando XGBoost.pkl...")
    with open(MODEL_PATH, "rb") as f:
        xgb_model = pickle.load(f)

    # ── Calcula confiança via embeddings já prontos ─────────────────────────
    print("Calculando confianca para os nao-DESCARTAVEL (embeddings.npy)...")
    X      = np.load(EMBEDDINGS)
    probas = xgb_model.predict_proba(X)
    preds  = probas.argmax(axis=1)
    confs  = probas.max(axis=1).round(4)

    confidence_df = pd.DataFrame({
        "commentId":  meta_df["commentId"].values,
        "xgb_pred":   [LABEL_MAP_REVERSE[int(p)] for p in preds],
        "confidence": confs,
    })

    # ── Concordância XGBoost × LLM (informativa) ────────────────────────────
    eval_df = meta_df[["commentId", "label_llm"]].merge(confidence_df, on="commentId")
    print(f"\n{'='*60}")
    print("  Concordancia XGBoost x LLM")
    print("  (ATENCAO: XGBoost foi treinado nesses dados)")
    print("  Use os resultados de CV do script 06 como avaliacao real.")
    print(f"{'='*60}")
    report = classification_report(
        eval_df["label_llm"],
        eval_df["xgb_pred"],
        labels=LABEL_ORDER,
        target_names=LABEL_ORDER,
    )
    print(report)

    cm = confusion_matrix(eval_df["label_llm"], eval_df["xgb_pred"], labels=LABEL_ORDER)
    header = "           " + "  ".join(f"{l:>10}" for l in LABEL_ORDER)
    print("Matriz de confusao:")
    print(header)
    for i, row in enumerate(cm):
        print(f"  {LABEL_ORDER[i]:>9}  " + "  ".join(f"{v:>10}" for v in row))

    # ── Monta dataset final ─────────────────────────────────────────────────
    print("\nMontando dataset unificado...")

    # Label LLM para todos os comentarios
    final_df = comments_df.merge(
        llm_df[["commentId", "label"]], on="commentId", how="inner"
    )

    # Confiança XGBoost para os nao-DESCARTAVEL (DESCARTAVEL fica com NaN)
    final_df = final_df.merge(
        confidence_df[["commentId", "confidence"]], on="commentId", how="left"
    )

    # ── Resumo ──────────────────────────────────────────────────────────────
    total = len(final_df)
    print(f"\n{'='*60}")
    print(f"  Dataset final: {total:,} comentarios")
    print(f"{'='*60}")

    print("\nDistribuicao de labels:")
    for lbl, cnt in final_df["label"].value_counts().items():
        print(f"  {lbl:<12} {cnt:>6}  ({cnt/total*100:.1f}%)")

    print("\nPor channel_type:")
    for ct, cnt in final_df["channel_type"].value_counts().items():
        print(f"  {ct:<14} {cnt:>6}  ({cnt/total*100:.1f}%)")

    sent_df = final_df[final_df["label"] != "DESCARTAVEL"]
    print(f"\nConfianca media por label (exclui DESCARTAVEL, n={len(sent_df):,}):")
    for lbl, mean_conf in sent_df.groupby("label")["confidence"].mean().round(3).items():
        print(f"  {lbl:<12} {mean_conf:.3f}")

    # Intensidade emocional (distribuicao)
    def intensidade(c):
        if pd.isna(c):  return None
        if c > 0.80:    return "ALTA"
        if c >= 0.60:   return "MEDIA"
        return "BAIXA"

    sent_df = sent_df.copy()
    sent_df["intensidade"] = sent_df["confidence"].apply(intensidade)
    print(f"\nDistribuicao de intensidade emocional (n={len(sent_df):,}):")
    for intens, cnt in sent_df["intensidade"].value_counts().items():
        print(f"  {intens:<6} {cnt:>6}  ({cnt/len(sent_df)*100:.1f}%)")

    # ── Salva ───────────────────────────────────────────────────────────────
    final_df.to_csv(OUT_FINAL, index=False, encoding="utf-8-sig")
    print(f"\nDataset final salvo: {OUT_FINAL}")
    print("Proximo passo: python scripts/08_analyze_results.py")


if __name__ == "__main__":
    main()
