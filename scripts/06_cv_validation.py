"""
06_cv_validation.py
--------------------
Avalia o pipeline BERTimbau + XGBoost via validação cruzada estratificada 5-fold.

Objetivo: validar academicamente a capacidade de generalização do modelo
sobre os embeddings gerados pelo script 05 (base completa de 34.491
comentários não-DESCARTAVEL rotulados pelo LLM).

Entrada:
    data/processed/embeddings.npy
    data/processed/labels.npy

Saídas:
    data/processed/models/resultados_cv.csv    — métricas por fold
    data/processed/models/relatorio_cv.txt     — relatório completo

Uso:
    python scripts/06_cv_validation.py

Requisitos:
    pip install scikit-learn xgboost pandas numpy
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

# ─── Configurações ────────────────────────────────────────────────────────────

K_FOLDS      = 5
RANDOM_STATE = 42

LABEL_NAMES = {0: "NEGATIVO", 1: "TANGENCIAL", 2: "POSITIVO"}

# ─── Caminhos ────────────────────────────────────────────────────────────────

ROOT       = Path(__file__).parent.parent
EMBEDDINGS = ROOT / "data" / "processed" / "embeddings.npy"
LABELS     = ROOT / "data" / "processed" / "labels.npy"
OUT_DIR    = ROOT / "data" / "processed" / "models"


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_lines = []

    def log(msg=""):
        print(msg)
        log_lines.append(msg)

    # ── Carrega dados ───────────────────────────────────────────────────────
    X = np.load(EMBEDDINGS)
    y = np.load(LABELS)

    log("=" * 60)
    log("  VALIDAÇÃO CRUZADA — TCC")
    log("=" * 60)
    log(f"Embeddings: {X.shape[0]:,} comentários × {X.shape[1]} dimensões")
    log(f"Validação : {K_FOLDS}-fold estratificado\n")

    log("Distribuição de classes:")
    for label_id, name in sorted(LABEL_NAMES.items()):
        count = (y == label_id).sum()
        log(f"  {name} ({label_id}): {count:,}  ({count/len(y)*100:.1f}%)")

    # ── Sample weights (balanceia classes raras como NEGATIVO) ──────────────
    sample_weights = compute_sample_weight(class_weight="balanced", y=y)

    # ── Modelo ──────────────────────────────────────────────────────────────
    clf = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        device="cuda",
        verbosity=0,
    )

    # ── Cross-validation ────────────────────────────────────────────────────
    log("\nExecutando cross-validation...")
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    scores = cross_validate(
        clf, X, y,
        cv=cv,
        scoring=["accuracy", "f1_macro", "f1_weighted"],
        params={"sample_weight": sample_weights},
        n_jobs=-1,
    )

    log("\n── Resultados por fold ─────────────────────────────────")
    for i in range(K_FOLDS):
        log(f"  Fold {i+1}: "
            f"Acurácia={scores['test_accuracy'][i]:.4f}  "
            f"F1-macro={scores['test_f1_macro'][i]:.4f}  "
            f"F1-weighted={scores['test_f1_weighted'][i]:.4f}")

    log("\n── Médias ± desvio padrão ──────────────────────────────")
    log(f"  Acurácia   : {scores['test_accuracy'].mean():.4f} "
        f"± {scores['test_accuracy'].std():.4f}")
    log(f"  F1-macro   : {scores['test_f1_macro'].mean():.4f} "
        f"± {scores['test_f1_macro'].std():.4f}")
    log(f"  F1-weighted: {scores['test_f1_weighted'].mean():.4f} "
        f"± {scores['test_f1_weighted'].std():.4f}")

    # ── Salva métricas por fold ─────────────────────────────────────────────
    results_df = pd.DataFrame({
        "fold":        range(1, K_FOLDS + 1),
        "acuracia":    scores["test_accuracy"].round(4),
        "f1_macro":    scores["test_f1_macro"].round(4),
        "f1_weighted": scores["test_f1_weighted"].round(4),
    })
    results_df.to_csv(OUT_DIR / "resultados_cv.csv", index=False, encoding="utf-8-sig")

    # ── Salva relatório ─────────────────────────────────────────────────────
    report_path = OUT_DIR / "relatorio_cv.txt"
    report_path.write_text("\n".join(log_lines), encoding="utf-8")
    log(f"\nRelatório salvo: {report_path}")
    log("Próximo passo: python scripts/06b_half_split_evaluation.py")


if __name__ == "__main__":
    main()
