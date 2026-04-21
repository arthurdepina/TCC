"""
04_sample_for_review.py
-----------------------
Seleciona 600 comentários para revisão humana e divide em dois arquivos
de 300 (um por revisora).

Estratégia de amostragem — por que não aleatório puro?
  Uma amostra aleatória pura teria ~54% positivos, ~24% negativos e ~22%
  descartáveis, refletindo o viés do dataset. Para uma revisão humana útil,
  queremos representação equilibrada entre rótulos e entre tipos de canal,
  e atenção especial aos casos que mais importam revisar.

Critérios de seleção (600 total):
  1. Casos de alta relevância por engajamento (likeCount > 50): 100
     — Comentários muito curtidos têm mais peso analítico no TCC.
  2. Amostra estratificada por label × channel_type: 400
     — 4 grupos (POSITIVO/NEGATIVO × profissional/amador), 100 cada.
     — DESCARTAVEL não precisa de revisão humana intensa.
  3. Comentários muito curtos com label POSITIVO ou NEGATIVO: 100
     — Textos < 15 chars são mais ambíguos (ex: "Amei", "Horrível").
     — Mais propensos a erro do modelo.

Os 600 são embaralhados e divididos aleatoriamente em dois arquivos de 300.

Saídas:
  data/processed/para_revisao_arthur.csv
  data/processed/para_revisao_ana.csv

Colunas de saída:
  commentId, text, channel_type, likeCount, video_title,
  label_llm, label_revisao
  (label_revisao começa vazia — preenchida manualmente pelas revisoras)
"""

import os
import pandas as pd
import numpy as np

# ─── Caminhos ────────────────────────────────────────────────────────────────

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS_CSV  = os.path.join(ROOT, "data", "processed", "llm_labeled_comments_v2.csv")
COMMENTS_CSV= os.path.join(ROOT, "data", "processed", "preprocessed_comments.csv")
OUT_ARTHUR  = os.path.join(ROOT, "data", "processed", "para_revisao_arthur.csv")
OUT_ANA     = os.path.join(ROOT, "data", "processed", "para_revisao_ana.csv")

RANDOM_SEED = 42
TOTAL       = 600
HALF        = TOTAL // 2

# ─── Carrega e une ────────────────────────────────────────────────────────────

labels   = pd.read_csv(LABELS_CSV)
comments = pd.read_csv(COMMENTS_CSV, usecols=[
    "commentId", "text", "channel_type", "likeCount",
    "video_title", "text_length"
])

df = labels.merge(comments, on="commentId")
print(f"Base disponivel: {len(df):,} comentarios rotulados")


# ─── Amostragem estratégica ───────────────────────────────────────────────────

selected_ids = set()
samples      = []

def add_sample(pool: pd.DataFrame, n: int, label: str) -> None:
    """Adiciona até n linhas do pool, excluindo IDs já selecionados."""
    pool = pool[~pool["commentId"].isin(selected_ids)]
    n    = min(n, len(pool))
    pick = pool.sample(n=n, random_state=RANDOM_SEED)
    samples.append(pick)
    selected_ids.update(pick["commentId"].tolist())
    print(f"  [{label}] {n} comentarios selecionados")


# 1. Alta relevância (likeCount > 50), apenas POSITIVO e NEGATIVO
high_like = df[(df["likeCount"] > 50) & (df["label_llm"].isin(["POSITIVO", "NEGATIVO"]))]
add_sample(high_like, 100, "Alta relevancia (likes > 50)")

# 2. Amostra estratificada: POSITIVO × profissional
pos_prof = df[(df["label_llm"] == "POSITIVO") & (df["channel_type"] == "profissional")]
add_sample(pos_prof, 100, "POSITIVO x profissional")

# 3. Amostra estratificada: POSITIVO × amador
pos_am = df[(df["label_llm"] == "POSITIVO") & (df["channel_type"] == "amador")]
add_sample(pos_am, 100, "POSITIVO x amador")

# 4. Amostra estratificada: NEGATIVO × profissional
neg_prof = df[(df["label_llm"] == "NEGATIVO") & (df["channel_type"] == "profissional")]
add_sample(neg_prof, 100, "NEGATIVO x profissional")

# 5. Amostra estratificada: NEGATIVO × amador
neg_am = df[(df["label_llm"] == "NEGATIVO") & (df["channel_type"] == "amador")]
add_sample(neg_am, 100, "NEGATIVO x amador")

# 6. Comentários curtos (< 15 chars) com label POSITIVO ou NEGATIVO
short = df[
    (df["text_length"] < 15) &
    (df["label_llm"].isin(["POSITIVO", "NEGATIVO"])) &
    (~df["commentId"].isin(selected_ids))
]
add_sample(short, 100, "Curtos (< 15 chars)")

# ─── Concatena, embaralha e divide ────────────────────────────────────────────

result = pd.concat(samples, ignore_index=True)
result = result.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

# Colunas finais para a revisora
output_cols = ["commentId", "text", "channel_type", "likeCount", "video_title", "label_llm"]
result = result[output_cols].copy()
result["label_revisao"] = ""   # coluna vazia para preenchimento manual

arthur = result.iloc[:HALF].reset_index(drop=True)
ana    = result.iloc[HALF:].reset_index(drop=True)

# ─── Salva ────────────────────────────────────────────────────────────────────

arthur.to_csv(OUT_ARTHUR, index=False, encoding="utf-8-sig")
ana.to_csv(OUT_ANA,    index=False, encoding="utf-8-sig")

print(f"\nTotal selecionado : {len(result)}")
print(f"Arthur            : {len(arthur)} comentarios -> {OUT_ARTHUR}")
print(f"Ana               : {len(ana)} comentarios -> {OUT_ANA}")

print("\nDistribuicao label_llm no conjunto de revisao:")
print(result["label_llm"].value_counts().to_string())
print("\nDistribuicao channel_type no conjunto de revisao:")
print(result["channel_type"].value_counts().to_string())
print("\nInstrucoes:")
print("  Preencha a coluna 'label_revisao' com: POSITIVO, NEGATIVO, VIVENCIAL ou DESCARTAVEL")
print("  Deixe em branco se concordar com o label_llm (so corrija o que discordar)")
