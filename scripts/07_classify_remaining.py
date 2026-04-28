"""
07_classify_remaining.py
-------------------------
Classifica os comentários sem rótulo usando o XGBoost treinado.
Ao final, gera um dataset unificado com todos os 54.485 comentários rotulados.

Fluxo:
    1. Identifica os ~33.299 comentários que NÃO foram rotulados pelo LLM
    2. Gera embeddings BERTimbau para esses comentários
    3. Usa o XGBoost treinado para prever o sentimento de cada um
    4. Une com os 21.186 já rotulados pelo LLM
    5. Salva o dataset final unificado

Mapeamento de rótulos do modelo:
    0 → NEGATIVO
    1 → TANGENCIAL
    2 → POSITIVO

Checkpoint automático: se interrompido, retoma de onde parou.

Entrada:
    data/processed/preprocessed_comments.csv
    data/processed/llm_labeled_comments_v3.csv
    data/processed/models/XGBoost.pkl

Saídas:
    data/processed/all_labeled_comments.csv   — dataset unificado final
    data/processed/remaining_checkpoint.json  — checkpoint de progresso

Uso:
    python scripts/07_classify_remaining.py

Requisitos:
    pip install transformers torch pandas numpy tqdm scikit-learn xgboost
"""

import json
import pickle
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

# ─── Configurações ────────────────────────────────────────────────────────────

MODEL_NAME   = "neuralmind/bert-base-portuguese-cased"
BATCH_SIZE   = 32
MAX_LENGTH   = 128
SAVE_EVERY   = 500   # salva checkpoint a cada N comentários processados

LABEL_MAP_REVERSE = {0: "NEGATIVO", 1: "TANGENCIAL", 2: "POSITIVO"}

# ─── Caminhos ────────────────────────────────────────────────────────────────

ROOT          = Path(__file__).parent.parent
COMMENTS_CSV  = ROOT / "data" / "processed" / "preprocessed_comments.csv"
LLM_CSV       = ROOT / "data" / "processed" / "llm_labeled_comments_v3.csv"
MODEL_PATH    = ROOT / "data" / "processed" / "models" / "XGBoost.pkl"
OUT_FINAL     = ROOT / "data" / "processed" / "all_labeled_comments.csv"
CHECKPOINT    = ROOT / "data" / "processed" / "remaining_checkpoint.json"
PARTIAL_CSV   = ROOT / "data" / "processed" / "remaining_predictions_partial.csv"


# ─── Checkpoint ──────────────────────────────────────────────────────────────

def load_checkpoint() -> set:
    if CHECKPOINT.exists():
        with open(CHECKPOINT, "r", encoding="utf-8") as f:
            return set(json.load(f).get("done_ids", []))
    return set()

def save_checkpoint(done_ids: set) -> None:
    with open(CHECKPOINT, "w", encoding="utf-8") as f:
        json.dump({"done_ids": list(done_ids)}, f)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    # ── Carrega bases ───────────────────────────────────────────────────────
    print("Carregando dados...")
    comments_df = pd.read_csv(COMMENTS_CSV, usecols=[
        "commentId", "text", "text_clean", "channel_type",
        "likeCount", "video_title"
    ])
    llm_df = pd.read_csv(LLM_CSV)  # commentId, label_llm

    # Identifica comentários sem rótulo LLM
    llm_ids = set(llm_df["commentId"])
    remaining_df = comments_df[~comments_df["commentId"].isin(llm_ids)].copy()
    remaining_df = remaining_df.reset_index(drop=True)

    print(f"Total na base         : {len(comments_df):,}")
    print(f"Rotulados pelo LLM    : {len(llm_df):,}")
    print(f"Pendentes (sem rótulo): {len(remaining_df):,}")

    # ── Checkpoint: pula já processados ────────────────────────────────────
    done_ids = load_checkpoint()
    pending_df = remaining_df[~remaining_df["commentId"].isin(done_ids)].copy()
    pending_df = pending_df.reset_index(drop=True)
    print(f"Já classificados      : {len(done_ids):,}")
    print(f"A classificar agora   : {len(pending_df):,}")

    if not pending_df.empty:
        # ── Carrega BERTimbau ───────────────────────────────────────────────
        print(f"\nCarregando {MODEL_NAME}...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Dispositivo: {device}")

        tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
        bert = BertModel.from_pretrained(MODEL_NAME)
        bert.eval()
        bert.to(device)

        # ── Carrega XGBoost ─────────────────────────────────────────────────
        print("Carregando XGBoost.pkl...")
        with open(MODEL_PATH, "rb") as f:
            xgb_model = pickle.load(f)

        # ── Carrega resultados parciais existentes ──────────────────────────
        if PARTIAL_CSV.exists():
            partial_results = pd.read_csv(PARTIAL_CSV).to_dict("records")
        else:
            partial_results = []

        # ── Classifica em batches ───────────────────────────────────────────
        texts = pending_df["text_clean"].fillna("").astype(str).tolist()
        ids   = pending_df["commentId"].tolist()
        n_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

        print(f"\nClassificando {len(texts):,} comentários em {n_batches} batches...\n")

        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Classificando"):
            batch_texts = texts[i : i + BATCH_SIZE]
            batch_ids   = ids[i : i + BATCH_SIZE]

            # Embedding BERTimbau
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}

            with torch.no_grad():
                output = bert(**encoded)

            cls_vecs = output.last_hidden_state[:, 0, :].cpu().numpy()

            # Predição XGBoost
            preds = xgb_model.predict(cls_vecs)

            for cid, pred in zip(batch_ids, preds):
                partial_results.append({
                    "commentId":  cid,
                    "label":      LABEL_MAP_REVERSE[int(pred)],
                    "label_source": "modelo",
                })
                done_ids.add(cid)

            # Salva checkpoint a cada SAVE_EVERY comentários
            if len(done_ids) % SAVE_EVERY < BATCH_SIZE:
                pd.DataFrame(partial_results).to_csv(
                    PARTIAL_CSV, index=False, encoding="utf-8-sig"
                )
                save_checkpoint(done_ids)

        # Salva resultado final das predições
        pd.DataFrame(partial_results).to_csv(
            PARTIAL_CSV, index=False, encoding="utf-8-sig"
        )
        save_checkpoint(done_ids)
        print(f"\nClassificação concluída: {len(partial_results):,} comentários.")

    # ── Monta dataset unificado ─────────────────────────────────────────────
    print("\nMontando dataset unificado...")

    # LLM-labeled: adiciona coluna label e label_source
    llm_enriched = llm_df.rename(columns={"label_llm": "label"}).copy()
    llm_enriched["label_source"] = "llm"

    # Predições do modelo
    model_preds = pd.read_csv(PARTIAL_CSV)

    # Une rótulos (LLM + modelo)
    all_labels = pd.concat([llm_enriched, model_preds], ignore_index=True)

    # Junta com metadados completos
    final_df = comments_df.merge(all_labels, on="commentId", how="inner")

    print(f"\nDataset unificado: {len(final_df):,} comentários")
    print("\nDistribuição de labels:")
    print(final_df["label"].value_counts().to_string())
    print("\nPor fonte de rótulo:")
    print(final_df["label_source"].value_counts().to_string())
    print("\nPor channel_type:")
    print(final_df["channel_type"].value_counts().to_string())

    final_df.to_csv(OUT_FINAL, index=False, encoding="utf-8-sig")
    print(f"\nDataset final salvo: {OUT_FINAL}")
    print("Próximo passo: python scripts/08_analyze_results.py")


if __name__ == "__main__":
    main()
