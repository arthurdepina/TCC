"""
08_credibility_heuristic.py
----------------------------
Extrai a métrica de confiança percebida para cada comentário da base final,
usando heurística de palavras-chave com tratamento de negação e desambiguação
via label_xgb.

Aplica-se às três classes de sentimento (POSITIVO, TANGENCIAL, NEGATIVO).

Lógica:
    1. Busca termos de ALTA credibilidade em text_clean
    2. Verifica negação (janela de 3 palavras antes do termo)
    3. Busca termos de BAIXA credibilidade
    4. Determina keyword_signal (ALTA / BAIXA / CONFLITO / NEUTRA)
    5. Resolve CONFLITO via label_xgb:
         POSITIVO   → ALTA
         NEGATIVO   → BAIXA
         TANGENCIAL → NEUTRA

Entrada:
    data/processed/final_labeled_dataset.csv

Saídas:
    data/processed/final_dataset.csv         — base completa + confianca_percebida
    data/processed/final_dataset_audit.csv   — + keyword_signal, keywords_alta,
                                                keywords_baixa, conflito (auditoria)

Uso:
    python scripts/08_credibility_heuristic.py

Requisitos:
    pip install pandas
"""

import sys
import pandas as pd
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ─── Configurações ────────────────────────────────────────────────────────────

NEGACAO_JANELA = 3
NEGACOES = {"não", "nunca", "jamais", "nem", "nenhum", "nada"}

TERMOS_ALTA = [
    "explicou", "explicação", "faz sentido", "confio", "confiança",
    "científico", "embasado", "comprovado", "especialista", "referência",
    "pesquisa", "estudo", "evidência", "recomendo", "recomenda",
    "correto", "verdade", "confiável", "fundamentado", "comprovou",
    "funciona", "ajudou", "ajuda muito", "esclareceu", "ficou claro",
    "bem explicado", "informação clara", "informação correta",
    "excelente conteúdo", "ótimo conteúdo",
]

TERMOS_BAIXA = [
    "errado", "equivocado", "pseudociência", "desinformação",
    "charlatão", "sem base", "enganação", "falso", "mentira",
    "perigoso", "irresponsável", "absurdo", "ridículo",
    "não funciona", "não confio", "não ajuda", "não faz sentido",
    "não é verdade", "não é correto", "não é confiável",
    "lixo", "besteira", "bobagem",
]

# ─── Caminhos ────────────────────────────────────────────────────────────────

ROOT      = Path(__file__).parent.parent
INPUT_CSV = ROOT / "data" / "processed" / "final_labeled_dataset.csv"
OUT_FINAL = ROOT / "data" / "processed" / "final_dataset.csv"
OUT_AUDIT = ROOT / "data" / "processed" / "final_dataset_audit.csv"


# ─── Detecção ─────────────────────────────────────────────────────────────────

def detectar_credibilidade(text: str, label_xgb: str) -> tuple:
    """
    Retorna (confianca_percebida, keyword_signal, keywords_alta, keywords_baixa).

    Para cada termo de ALTA, verifica se há palavra de negação nas
    NEGACAO_JANELA palavras anteriores. Se houver, o termo é tratado
    como sinal de BAIXA em vez de ALTA.
    """
    t = str(text).lower()

    # ── Termos de ALTA (com verificação de negação) ──────────────────────────
    alta_pos, alta_negados = [], []
    for termo in TERMOS_ALTA:
        idx = t.find(termo)
        if idx == -1:
            continue
        palavras_antes = t[:idx].split()[-NEGACAO_JANELA:]
        if any(neg in palavras_antes for neg in NEGACOES):
            alta_negados.append(termo)
        else:
            alta_pos.append(termo)

    # ── Termos de BAIXA + termos de ALTA negados ─────────────────────────────
    baixa = [termo for termo in TERMOS_BAIXA if termo in t] + alta_negados

    # ── keyword_signal ────────────────────────────────────────────────────────
    tem_alta  = len(alta_pos) > 0
    tem_baixa = len(baixa) > 0

    if tem_alta and not tem_baixa:
        keyword_signal = "ALTA"
    elif tem_baixa and not tem_alta:
        keyword_signal = "BAIXA"
    elif tem_alta and tem_baixa:
        keyword_signal = "CONFLITO"
    else:
        keyword_signal = "NEUTRA"

    # ── Resolve CONFLITO via label_xgb ────────────────────────────────────────
    if keyword_signal == "CONFLITO":
        if label_xgb == "POSITIVO":
            confianca = "ALTA"
        elif label_xgb == "NEGATIVO":
            confianca = "BAIXA"
        else:
            confianca = "NEUTRA"
    else:
        confianca = keyword_signal

    return confianca, keyword_signal, ";".join(alta_pos), ";".join(baixa)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("Carregando base...")
    df = pd.read_csv(INPUT_CSV)
    print(f"  {len(df):,} comentários")

    # ── Aplica heurística ─────────────────────────────────────────────────────
    print("\nCalculando confiança percebida...")
    resultados = df.apply(
        lambda row: detectar_credibilidade(row["text_clean"], row["label_xgb"]),
        axis=1,
        result_type="expand",
    )
    resultados.columns = ["confianca_percebida", "keyword_signal", "keywords_alta", "keywords_baixa"]

    df_out = pd.concat([df, resultados], axis=1)
    df_out["conflito"] = df_out["keyword_signal"] == "CONFLITO"

    # ── Resumo ────────────────────────────────────────────────────────────────
    total = len(df_out)
    print(f"\n{'='*60}")
    print(f"  Distribuição — confianca_percebida  (n={total:,})")
    print(f"{'='*60}")
    for val, cnt in df_out["confianca_percebida"].value_counts().items():
        print(f"  {val:<8} {cnt:>6}  ({cnt/total*100:.1f}%)")

    n_conflito = df_out["conflito"].sum()
    print(f"\nConflitos desambiguados via label_xgb: {n_conflito:,} ({n_conflito/total*100:.1f}%)")

    print(f"\nCruzamento label_xgb × confianca_percebida:")
    cross = pd.crosstab(df_out["label_xgb"], df_out["confianca_percebida"])
    print(cross.to_string())

    # ── Salva dataset final (todas as colunas + confianca_percebida) ──────────
    cols_final = [c for c in df_out.columns
                  if c not in ["keyword_signal", "keywords_alta", "keywords_baixa", "conflito"]]
    df_out[cols_final].to_csv(OUT_FINAL, index=False, encoding="utf-8-sig")
    print(f"\nDataset final salvo   : {OUT_FINAL}")

    # ── Salva arquivo de auditoria ────────────────────────────────────────────
    audit_cols = [
        "commentId", "text_clean", "label_xgb",
        "confianca_percebida", "keyword_signal",
        "keywords_alta", "keywords_baixa", "conflito",
    ]
    df_out[audit_cols].to_csv(OUT_AUDIT, index=False, encoding="utf-8-sig")
    print(f"Arquivo de auditoria  : {OUT_AUDIT}")
    print("\nPróximo passo: python scripts/09_analyze_results.py")


if __name__ == "__main__":
    main()
