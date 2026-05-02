"""
06b_half_split_evaluation.py
-----------------------------
Treina o XGBoost na primeira metade da base e avalia na segunda.

Objetivo: comparar os rotulos gerados pelo XGBoost com os rotulos LLM
na segunda metade — avaliacao genuina, pois o modelo nao viu esses dados.

A divisao respeita a ordem original do preprocessed_comments.csv:
    Primeira metade : linhas 0-21185  (rotuladas primeiro pelo LLM)
    Segunda metade  : linhas 21186+   (rotuladas depois pelo LLM)

Os embeddings.npy existentes (34.491 x 768) sao divididos por essa
logica sem precisar rodar o BERTimbau novamente.

Alem da avaliacao, o script gera o label e a intensidade emocional
do XGBoost para cada comentario da segunda metade:
    confianca > 0.80  -> intensidade ALTA
    confianca 0.60-0.80 -> intensidade MEDIA
    confianca < 0.60  -> intensidade BAIXA

Entradas:
    data/processed/preprocessed_comments.csv
    data/processed/llm_labeled_comments_v3.csv
    data/processed/embeddings.npy
    data/processed/embedding_meta.csv

Saidas (nao sobrescreve nada existente):
    data/processed/models/XGBoost_half1.pkl
    data/processed/models/resultados_half_split.txt
    data/processed/second_half_predictions.csv

Colunas de second_half_predictions.csv:
    commentId, label_llm, label_xgb, confidence, intensidade, concorda

Uso:
    python scripts/06b_half_split_evaluation.py
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ─── Configurações ────────────────────────────────────────────────────────────

FIRST_HALF_LIMIT = 21_186      # linhas 0..21185 do preprocessed_comments.csv
RANDOM_STATE     = 42

LABEL_MAP = {"NEGATIVO": 0, "TANGENCIAL": 1, "POSITIVO": 2}
LABEL_MAP_REVERSE = {v: k for k, v in LABEL_MAP.items()}
LABEL_ORDER = ["NEGATIVO", "TANGENCIAL", "POSITIVO"]

# ─── Caminhos ────────────────────────────────────────────────────────────────

ROOT         = Path(__file__).parent.parent
COMMENTS_CSV = ROOT / "data" / "processed" / "preprocessed_comments.csv"
LLM_CSV      = ROOT / "data" / "processed" / "llm_labeled_comments_v3.csv"
EMBEDDINGS   = ROOT / "data" / "processed" / "embeddings.npy"
META_CSV     = ROOT / "data" / "processed" / "embedding_meta.csv"
OUT_MODEL    = ROOT / "data" / "processed" / "models" / "XGBoost_half1.pkl"
OUT_REPORT   = ROOT / "data" / "processed" / "models" / "resultados_half_split.txt"
OUT_PREDS    = ROOT / "data" / "processed" / "second_half_predictions.csv"


# ─── Intensidade ─────────────────────────────────────────────────────────────

def intensidade(conf: float) -> str:
    if conf > 0.80:  return "ALTA"
    if conf >= 0.60: return "MEDIA"
    return "BAIXA"


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    log_lines = []

    def log(msg=""):
        print(msg)
        log_lines.append(msg)

    # ── Identifica IDs de cada metade ───────────────────────────────────────
    log("Carregando dados...")
    comments_df = pd.read_csv(COMMENTS_CSV, usecols=["commentId"])
    first_half_ids  = set(comments_df.head(FIRST_HALF_LIMIT)["commentId"])
    second_half_ids = set(comments_df.iloc[FIRST_HALF_LIMIT:]["commentId"])
    log(f"  1a metade : {len(first_half_ids):,} comentarios")
    log(f"  2a metade : {len(second_half_ids):,} comentarios")

    # ── Carrega embeddings e metadados ──────────────────────────────────────
    meta_df = pd.read_csv(META_CSV)   # commentId, label_llm, label(int), ...
    X_all   = np.load(EMBEDDINGS)     # (34.491, 768)

    log(f"\nEmbeddings carregados: {X_all.shape}")

    # ── Divide embeddings por metade ─────────────────────────────────────────
    mask_first  = meta_df["commentId"].isin(first_half_ids).values
    mask_second = meta_df["commentId"].isin(second_half_ids).values

    X_train  = X_all[mask_first]
    y_train  = meta_df.loc[mask_first,  "label"].values
    meta_train = meta_df[mask_first].reset_index(drop=True)

    X_test   = X_all[mask_second]
    y_test   = meta_df.loc[mask_second, "label"].values
    meta_test  = meta_df[mask_second].reset_index(drop=True)

    log(f"\nDivisao (nao-DESCARTAVEL):")
    log(f"  Treino (1a metade): {len(X_train):,} comentarios")
    for name, idx in sorted(LABEL_MAP.items(), key=lambda x: x[1]):
        cnt = (y_train == idx).sum()
        log(f"    {name} ({idx}): {cnt:,}  ({cnt/len(y_train)*100:.1f}%)")
    log(f"  Teste  (2a metade): {len(X_test):,} comentarios")
    for name, idx in sorted(LABEL_MAP.items(), key=lambda x: x[1]):
        cnt = (y_test == idx).sum()
        log(f"    {name} ({idx}): {cnt:,}  ({cnt/len(y_test)*100:.1f}%)")

    # ── Treina XGBoost na primeira metade ────────────────────────────────────
    log("\nTreinando XGBoost na primeira metade...")
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    clf = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        device="cuda",
        verbosity=0,
    )
    clf.fit(X_train, y_train, sample_weight=sample_weights)

    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_MODEL, "wb") as f:
        pickle.dump(clf, f)
    log(f"Modelo salvo: {OUT_MODEL}")

    # ── Avalia na segunda metade ─────────────────────────────────────────────
    log("\n" + "=" * 60)
    log("  AVALIACAO — XGBoost (treino: 1a metade) x LLM (2a metade)")
    log("  Esta e a comparacao genuina: modelo nunca viu esses dados.")
    log("=" * 60)

    probas_test = clf.predict_proba(X_test)
    preds_test  = probas_test.argmax(axis=1)
    confs_test  = probas_test.max(axis=1).round(4)

    report = classification_report(
        y_test, preds_test,
        labels=list(range(len(LABEL_ORDER))),
        target_names=LABEL_ORDER,
    )
    log("\n" + report)

    cm = confusion_matrix(y_test, preds_test, labels=list(range(len(LABEL_ORDER))))
    log("Matriz de confusao (linhas=LLM, colunas=XGBoost):")
    header = "           " + "  ".join(f"{l:>10}" for l in LABEL_ORDER)
    log(header)
    for i, row in enumerate(cm):
        log(f"  {LABEL_ORDER[i]:>9}  " + "  ".join(f"{v:>10}" for v in row))

    # Concordancia geral
    concorda = (preds_test == y_test)
    log(f"\nConcordancia geral: {concorda.sum():,} / {len(y_test):,}  ({concorda.mean()*100:.1f}%)")

    # ── Distribuicao de intensidade (segunda metade) ─────────────────────────
    log("\nDistribuicao de intensidade (segunda metade — label XGBoost):")
    intens_vals = [intensidade(c) for c in confs_test]
    for nivel in ["ALTA", "MEDIA", "BAIXA"]:
        cnt = intens_vals.count(nivel)
        log(f"  {nivel:<6} {cnt:>6}  ({cnt/len(intens_vals)*100:.1f}%)")

    log("\nConfianca media por label XGBoost:")
    for idx, name in LABEL_MAP_REVERSE.items():
        mask = preds_test == idx
        if mask.sum() > 0:
            log(f"  {name:<12} {confs_test[mask].mean():.3f}")

    # ── Salva predicoes da segunda metade ────────────────────────────────────
    preds_df = pd.DataFrame({
        "commentId":  meta_test["commentId"].values,
        "label_llm":  [LABEL_MAP_REVERSE[int(v)] for v in y_test],
        "label_xgb":  [LABEL_MAP_REVERSE[int(p)] for p in preds_test],
        "confidence": confs_test,
        "intensidade": intens_vals,
        "concorda":   concorda,
    })
    preds_df.to_csv(OUT_PREDS, index=False, encoding="utf-8-sig")
    log(f"\nPredicoes salvas: {OUT_PREDS}")

    # ── Salva relatorio ──────────────────────────────────────────────────────
    OUT_REPORT.write_text("\n".join(log_lines), encoding="utf-8")
    log(f"Relatorio salvo: {OUT_REPORT}")


if __name__ == "__main__":
    main()
