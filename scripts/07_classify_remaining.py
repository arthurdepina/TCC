"""
07_classify_remaining.py
-------------------------
Classifica os comentários sem rótulo e gera o dataset unificado final.

Além do rótulo, salva o score de CONFIANÇA de cada predição (probabilidade
máxima do XGBoost), que será usado no script 08 para calcular a INTENSIDADE
emocional conforme definido no TCC:
    confiança > 0.80  → intensidade ALTA
    confiança 0.60–0.80 → intensidade MÉDIA
    confiança < 0.60  → intensidade BAIXA

Para os comentários LLM-labeled (base de treino), a confiança é calculada
passando seus embeddings já existentes (embeddings.npy) pelo XGBoost.
Para os comentários restantes, gera embeddings via BERTimbau e depois prediz.

Fluxo:
    1. Calcula confiança para os 17.772 LLM-labeled (usa embeddings.npy já prontos)
    2. Gera embeddings BERTimbau para os ~33.299 sem rótulo
    3. Prediz rótulo + confiança para os ~33.299 via XGBoost
    4. Une tudo e salva dataset unificado com sentimento + confiança

Checkpoint automático: se interrompido, retoma de onde parou.

Entradas:
    data/processed/preprocessed_comments.csv
    data/processed/llm_labeled_comments_v3.csv
    data/processed/embeddings.npy          ← embeddings da base de treino (script 05)
    data/processed/labels.npy
    data/processed/embedding_meta.csv
    data/processed/models/XGBoost.pkl

Saídas:
    data/processed/all_labeled_comments.csv     ← dataset unificado final
    data/processed/remaining_checkpoint.json    ← checkpoint de progresso

Colunas do dataset final:
    commentId, text, text_clean, channel_type, likeCount, video_title,
    label, confidence, label_source

Uso:
    python scripts/07_classify_remaining.py
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

MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
BATCH_SIZE = 32
MAX_LENGTH = 128
SAVE_EVERY = 500

LABEL_MAP_REVERSE = {0: "NEGATIVO", 1: "TANGENCIAL", 2: "POSITIVO"}

# ─── Caminhos ────────────────────────────────────────────────────────────────

ROOT         = Path(__file__).parent.parent
COMMENTS_CSV = ROOT / "data" / "processed" / "preprocessed_comments.csv"
LLM_CSV      = ROOT / "data" / "processed" / "llm_labeled_comments_v3.csv"
EMBEDDINGS   = ROOT / "data" / "processed" / "embeddings.npy"
META_CSV     = ROOT / "data" / "processed" / "embedding_meta.csv"
MODEL_PATH   = ROOT / "data" / "processed" / "models" / "XGBoost.pkl"
OUT_FINAL    = ROOT / "data" / "processed" / "all_labeled_comments.csv"
CHECKPOINT   = ROOT / "data" / "processed" / "remaining_checkpoint.json"
PARTIAL_CSV  = ROOT / "data" / "processed" / "remaining_predictions_partial.csv"


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
    llm_df  = pd.read_csv(LLM_CSV)   # commentId, label_llm
    meta_df = pd.read_csv(META_CSV)  # commentId, label_llm, label(int), channel_type...

    # Carrega XGBoost
    print("Carregando XGBoost.pkl...")
    with open(MODEL_PATH, "rb") as f:
        xgb_model = pickle.load(f)

    # ── Passo 1: confiança para os comentários LLM-labeled ─────────────────
    # Usa embeddings.npy já prontos (gerados no script 05) — não repassa pelo BERTimbau
    print("\nCalculando confiança para comentários LLM-labeled (embeddings prontos)...")
    X_train = np.load(EMBEDDINGS)
    probas_train = xgb_model.predict_proba(X_train)
    confidence_train = probas_train.max(axis=1)  # probabilidade da classe vencedora

    # Monta DataFrame com confiança para cada commentId da base de treino
    llm_confidence_df = pd.DataFrame({
        "commentId":  meta_df["commentId"].values,
        "confidence": confidence_train.round(4),
    })

    # Junta com labels LLM (mantém o rótulo original do LLM, não o do XGBoost)
    llm_enriched = llm_df.rename(columns={"label_llm": "label"}).merge(
        llm_confidence_df, on="commentId", how="left"
    )
    # DESCARTAVEL não participa da análise de intensidade — confiança será NaN mantida
    llm_enriched["confidence"] = llm_enriched.apply(
        lambda r: r["confidence"] if r["label"] != "DESCARTAVEL" else None, axis=1
    )
    llm_enriched["label_source"] = "llm"

    print(f"  LLM-labeled processados: {len(llm_enriched):,}")

    # ── Passo 2: classifica os comentários sem rótulo ───────────────────────
    llm_ids       = set(llm_df["commentId"])
    remaining_df  = comments_df[~comments_df["commentId"].isin(llm_ids)].copy()
    remaining_df  = remaining_df.reset_index(drop=True)

    print(f"\nComentários sem rótulo: {len(remaining_df):,}")

    done_ids    = load_checkpoint()
    pending_df  = remaining_df[~remaining_df["commentId"].isin(done_ids)].copy()
    pending_df  = pending_df.reset_index(drop=True)
    print(f"Já classificados      : {len(done_ids):,}")
    print(f"A classificar agora   : {len(pending_df):,}")

    if not pending_df.empty:
        # Carrega BERTimbau
        print(f"\nCarregando {MODEL_NAME}...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Dispositivo: {device}")

        tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
        bert = BertModel.from_pretrained(MODEL_NAME)
        bert.eval()
        bert.to(device)

        # Carrega resultados parciais existentes
        if PARTIAL_CSV.exists():
            partial_results = pd.read_csv(PARTIAL_CSV).to_dict("records")
        else:
            partial_results = []

        texts = pending_df["text_clean"].fillna("").astype(str).tolist()
        ids   = pending_df["commentId"].tolist()

        print(f"\nClassificando {len(texts):,} comentários...\n")

        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="BERTimbau + XGBoost"):
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

            # Predição XGBoost com probabilidades
            probas = xgb_model.predict_proba(cls_vecs)
            preds  = probas.argmax(axis=1)
            confs  = probas.max(axis=1)

            for cid, pred, conf in zip(batch_ids, preds, confs):
                partial_results.append({
                    "commentId":    cid,
                    "label":        LABEL_MAP_REVERSE[int(pred)],
                    "confidence":   round(float(conf), 4),
                    "label_source": "modelo",
                })
                done_ids.add(cid)

            # Checkpoint a cada SAVE_EVERY comentários
            if len(done_ids) % SAVE_EVERY < BATCH_SIZE:
                pd.DataFrame(partial_results).to_csv(
                    PARTIAL_CSV, index=False, encoding="utf-8-sig"
                )
                save_checkpoint(done_ids)

        # Salva predições finais
        pd.DataFrame(partial_results).to_csv(
            PARTIAL_CSV, index=False, encoding="utf-8-sig"
        )
        save_checkpoint(done_ids)
        print(f"\nClassificação concluída: {len(partial_results):,} comentários.")

    # ── Passo 3: monta dataset unificado ────────────────────────────────────
    print("\nMontando dataset unificado...")

    model_preds = pd.read_csv(PARTIAL_CSV)
    all_labels  = pd.concat([
        llm_enriched[["commentId", "label", "confidence", "label_source"]],
        model_preds[["commentId", "label", "confidence", "label_source"]],
    ], ignore_index=True)

    final_df = comments_df.merge(all_labels, on="commentId", how="inner")

    # ── Resumo ──────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Dataset unificado: {len(final_df):,} comentários")
    print(f"{'='*55}")

    print("\nDistribuição de labels:")
    print(final_df["label"].value_counts().to_string())

    print("\nPor fonte de rótulo:")
    print(final_df["label_source"].value_counts().to_string())

    print("\nPor channel_type:")
    print(final_df["channel_type"].value_counts().to_string())

    print("\nConfiança média por label (excluindo DESCARTAVEL):")
    sentimento_df = final_df[final_df["label"] != "DESCARTAVEL"]
    print(sentimento_df.groupby("label")["confidence"].mean().round(3).to_string())

    final_df.to_csv(OUT_FINAL, index=False, encoding="utf-8-sig")
    print(f"\nDataset final salvo: {OUT_FINAL}")
    print("Próximo passo: python scripts/08_analyze_results.py")


if __name__ == "__main__":
    main()
