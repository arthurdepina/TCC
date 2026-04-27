"""
05_generate_embeddings.py
--------------------------
Gera embeddings BERTimbau para os comentários rotulados (base de treinamento).

O BERTimbau é usado como feature extractor com pesos congelados: cada comentário
é transformado no vetor [CLS] da última camada oculta (768 dimensões). Esse vetor
representa o sentido completo do comentário e será a entrada dos classificadores.

Mapeamento de rótulos:
    NEGATIVO    → 0
    TANGENCIAL  → 1
    POSITIVO    → 2
    DESCARTAVEL → excluído do treinamento

Entrada:
    data/processed/llm_labeled_comments_v3.csv
    data/processed/preprocessed_comments.csv

Saídas:
    data/processed/embeddings.npy       — matriz (N, 768) de float32
    data/processed/labels.npy           — vetor (N,) de inteiros
    data/processed/embedding_meta.csv   — commentId, label, channel_type, likeCount

Uso:
    python scripts/05_generate_embeddings.py

Requisitos:
    pip install transformers torch pandas numpy tqdm
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

# ─── Configurações ────────────────────────────────────────────────────────────

MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
BATCH_SIZE = 32    # reduzir para 16 se der erro de memória (RAM insuficiente)
MAX_LENGTH = 128   # cobre >99% dos comentários; trunca os raros muito longos

LABEL_MAP = {
    "NEGATIVO":   0,
    "TANGENCIAL": 1,
    "POSITIVO":   2,
    # DESCARTAVEL é excluído
}

# ─── Caminhos ────────────────────────────────────────────────────────────────

ROOT          = Path(__file__).parent.parent
LABELS_CSV    = ROOT / "data" / "processed" / "llm_labeled_comments_v3.csv"
COMMENTS_CSV  = ROOT / "data" / "processed" / "preprocessed_comments.csv"
OUT_EMBEDDINGS = ROOT / "data" / "processed" / "embeddings.npy"
OUT_LABELS    = ROOT / "data" / "processed" / "labels.npy"
OUT_META      = ROOT / "data" / "processed" / "embedding_meta.csv"


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    # ── Carrega e une datasets ──────────────────────────────────────────────
    print("Carregando dados...")
    labels_df = pd.read_csv(LABELS_CSV)
    comments_df = pd.read_csv(COMMENTS_CSV, usecols=[
        "commentId", "text_clean", "channel_type", "likeCount", "video_title"
    ])

    df = labels_df.merge(comments_df, on="commentId", how="inner")

    # Filtra somente rótulos de sentimento (exclui DESCARTAVEL)
    df = df[df["label_llm"].isin(LABEL_MAP)].copy()
    df["label"] = df["label_llm"].map(LABEL_MAP).astype(int)
    df["text_clean"] = df["text_clean"].fillna("").astype(str)
    df = df.reset_index(drop=True)

    print(f"Comentários para embedding: {len(df):,}")
    print("Distribuição:")
    for name, idx in sorted(LABEL_MAP.items(), key=lambda x: x[1]):
        count = (df["label"] == idx).sum()
        print(f"  {name} ({idx}): {count:,} ({count/len(df)*100:.1f}%)")

    # ── Carrega BERTimbau ───────────────────────────────────────────────────
    print(f"\nCarregando {MODEL_NAME}...")
    print("(primeira execução faz download de ~440MB — pode demorar)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertModel.from_pretrained(MODEL_NAME)
    model.eval()
    model.to(device)

    # ── Gera embeddings em batches ──────────────────────────────────────────
    texts = df["text_clean"].tolist()
    n_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    all_embeddings = []

    print(f"\nGerando embeddings — {n_batches} batches de até {BATCH_SIZE}...")

    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="BERTimbau"):
        batch = texts[i : i + BATCH_SIZE]

        encoded = tokenizer(
            batch,
            padding=True,       # padding dinâmico até o maior do batch
            truncation=True,    # trunca comentários muito longos
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            output = model(**encoded)

        # Vetor CLS: posição 0 da última camada oculta — resume o comentário inteiro
        cls_vecs = output.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(cls_vecs)

    # ── Salva resultados ────────────────────────────────────────────────────
    embeddings = np.vstack(all_embeddings).astype(np.float32)
    labels = df["label"].values.astype(np.int32)

    print(f"\nShape dos embeddings: {embeddings.shape}")  # (N, 768)

    np.save(OUT_EMBEDDINGS, embeddings)
    np.save(OUT_LABELS, labels)

    meta = df[["commentId", "label_llm", "label", "channel_type",
               "likeCount", "video_title"]].copy()
    meta.to_csv(OUT_META, index=False, encoding="utf-8-sig")

    print(f"Embeddings salvos  : {OUT_EMBEDDINGS}")
    print(f"Labels salvos      : {OUT_LABELS}")
    print(f"Metadados salvos   : {OUT_META}")
    print("\nPróximo passo: python scripts/06_train_classifier.py")


if __name__ == "__main__":
    main()
