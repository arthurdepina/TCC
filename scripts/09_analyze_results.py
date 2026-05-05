"""
09_analyze_results.py
---------------------
Analisa o dataset final e responde as 6 perguntas de pesquisa do TCC.
Gera arquivos CSV de resultado para cada análise.

Saídas (em data/processed/results/):
  p1_engajamento.csv              P1: sentimentos em comentários com likeCount >= 10
  p2_p6_sentimentos_base.csv      P2 e P6: sentimentos na base completa por grupo
  p3_intensidade_profissional.csv P3: intensidade emocional — canais profissionais
  p3_intensidade_amador.csv       P3: intensidade emocional — canais amadores
  p4_confianca.csv                P4: confiança percebida (ALTA/BAIXA) por grupo
  p5_concorda.csv                 P5: concordância geral LLM vs XGBoost
  p5_matriz_confusao.csv          P5: matriz de confusão LLM × XGBoost
  p5_kappa.csv                    P5: Cohen's Kappa
  estatisticas.csv                Todos os testes estatísticos consolidados

Estrutura dos CSVs de análise:
  - Coluna "visao": "percentual" ou "absoluto"
  - Percentuais calculados sobre o total do grupo no contexto de cada filtro

Uso:
  python scripts/09_analyze_results.py

Requisitos:
  pip install pandas scipy scikit-learn
"""

import sys
import pandas as pd
from pathlib import Path
from scipy.stats import chi2_contingency
from sklearn.metrics import cohen_kappa_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT    = Path(__file__).parent.parent
INPUT   = ROOT / "data" / "processed" / "final_dataset.csv"
OUT_DIR = ROOT / "data" / "processed" / "results"

GRUPOS    = ["profissional", "amador"]
LABELS    = ["POSITIVO", "NEGATIVO", "TANGENCIAL"]
LABEL_PT  = {"POSITIVO": "Positivo", "NEGATIVO": "Negativo", "TANGENCIAL": "Tangencial"}
INTENS    = ["ALTA", "MEDIA", "BAIXA"]
INTENS_PT = {"ALTA": "Alta intensidade", "MEDIA": "Média intensidade", "BAIXA": "Baixa intensidade"}

_stats = []  # acumula resultados de todos os testes estatísticos


# ─── Utilitários ─────────────────────────────────────────────────────────────

def _chi2(contingency_df, descricao):
    """Executa chi-quadrado, armazena resultado e retorna string descritiva."""
    chi2_val, p, dof, _ = chi2_contingency(contingency_df)
    sig = "Sim" if p < 0.05 else "Não"
    _stats.append({
        "Análise": descricao,
        "Teste": "Chi-quadrado",
        "Estatística": round(chi2_val, 4),
        "p-valor": round(p, 6),
        "Graus de liberdade": dof,
        "Diferença significativa (α=0.05)": sig,
    })
    return (
        f"Chi² = {chi2_val:.4f}, p = {p:.6f}, g.l. = {dof} "
        f"— {'Significativa' if sig == 'Sim' else 'Não significativa'} (α=0.05)"
    )


def _label_rows(df, labels):
    """
    Gera linhas percentual+absoluto de distribuição de label_xgb por grupo.
    Denominador = total de comentários do grupo (todos os rótulos incluídos).
    """
    rows = []
    for visao in ("percentual", "absoluto"):
        for label in labels:
            row = {"visao": visao, "rotulo": LABEL_PT[label]}
            for grupo in GRUPOS:
                g = df[df["channel_type"] == grupo]
                total = len(g)
                count = int((g["label_xgb"] == label).sum())
                row[grupo.capitalize()] = (
                    round(count / total * 100, 2) if visao == "percentual" and total > 0
                    else count
                )
            rows.append(row)
    return rows


def _save(rows, path):
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


# ─── P1: Comentários engajados (likeCount >= 10) ─────────────────────────────

def analise_p1(df):
    print("\n[P1] Comentários engajados (likeCount >= 10)...")
    filtrado = df[df["likeCount"] >= 10]

    rows = _label_rows(filtrado, LABELS)
    _save(rows, OUT_DIR / "p1_engajamento.csv")

    contingency = pd.crosstab(filtrado["channel_type"], filtrado["label_xgb"])
    chi2_str = _chi2(contingency, "P1 — Sentimentos em comentários engajados (likeCount >= 10)")

    print(f"  Salvo: p1_engajamento.csv")
    print(f"  Total (likeCount >= 10): {len(filtrado)}", end="")
    for g in GRUPOS:
        print(f"  |  {g.capitalize()}: {len(filtrado[filtrado['channel_type'] == g])}", end="")
    print(f"\n  {chi2_str}")


# ─── P2 + P6: Sentimentos na base completa ───────────────────────────────────

def analise_p2_p6(df):
    print("\n[P2/P6] Sentimentos na base completa...")

    rows = _label_rows(df, LABELS)
    _save(rows, OUT_DIR / "p2_p6_sentimentos_base.csv")

    contingency = pd.crosstab(df["channel_type"], df["label_xgb"])
    chi2_str = _chi2(contingency, "P2/P6 — Distribuição de sentimentos na base completa")

    print(f"  Salvo: p2_p6_sentimentos_base.csv  (n={len(df):,})")
    print(f"  {chi2_str}")


# ─── P3: Intensidade emocional (exclui TANGENCIAL) ───────────────────────────

def analise_p3(df):
    print("\n[P3] Intensidade emocional (POSITIVO e NEGATIVO apenas)...")

    base = df[df["label_xgb"].isin(["POSITIVO", "NEGATIVO"])].copy()

    contingency = pd.crosstab(base["channel_type"], base["intensidade"])
    chi2_str = _chi2(contingency, "P3 — Intensidade emocional por grupo")
    print(f"  {chi2_str}")

    for grupo in GRUPOS:
        g     = base[base["channel_type"] == grupo]
        g_pos = g[g["label_xgb"] == "POSITIVO"]
        g_neg = g[g["label_xgb"] == "NEGATIVO"]

        rows = []
        for visao in ("percentual", "absoluto"):
            for intens in INTENS:
                row = {"visao": visao, "intensidade": INTENS_PT[intens]}
                for col_name, subset in [("Total", g), ("Positivo", g_pos), ("Negativo", g_neg)]:
                    total = len(subset)
                    count = int((subset["intensidade"] == intens).sum())
                    row[col_name] = (
                        round(count / total * 100, 2) if visao == "percentual" and total > 0
                        else count
                    )
                rows.append(row)

        _save(rows, OUT_DIR / f"p3_intensidade_{grupo}.csv")
        print(
            f"  Salvo: p3_intensidade_{grupo}.csv"
            f"  (n={len(g)} | pos={len(g_pos)} | neg={len(g_neg)})"
        )


# ─── P4: Confiança percebida (ALTA / BAIXA) ──────────────────────────────────

def analise_p4(df):
    print("\n[P4] Confiança percebida...")

    filtrado = df[df["confianca_percebida"].isin(["ALTA", "BAIXA"])]
    confs    = ["ALTA", "BAIXA"]
    conf_pt  = {"ALTA": "Alta confiança percebida", "BAIXA": "Baixa confiança percebida"}

    rows = []
    for visao in ("percentual", "absoluto"):
        for conf in confs:
            row = {"visao": visao, "confianca": conf_pt[conf]}
            for grupo in GRUPOS:
                g = filtrado[filtrado["channel_type"] == grupo]
                total = len(g)
                count = int((g["confianca_percebida"] == conf).sum())
                row[grupo.capitalize()] = (
                    round(count / total * 100, 2) if visao == "percentual" and total > 0
                    else count
                )
            rows.append(row)

    _save(rows, OUT_DIR / "p4_confianca.csv")

    contingency = pd.crosstab(filtrado["channel_type"], filtrado["confianca_percebida"])
    chi2_str = _chi2(contingency, "P4 — Confiança percebida por grupo")

    print(f"  Salvo: p4_confianca.csv  (n={len(filtrado)} — NEUTRA excluída)")
    print(f"  {chi2_str}")


# ─── P5: Concordância LLM vs XGBoost ─────────────────────────────────────────

def analise_p5(df):
    print("\n[P5] Concordância LLM vs XGBoost...")

    total = len(df)
    counts = df["concorda"].value_counts()

    # 1. Concorda
    rows_concorda = [
        {
            "Concordância": "Concordam (True)",
            "Quantidade": int(counts.get(True, 0)),
            "Percentual (%)": round(counts.get(True, 0) / total * 100, 2),
        },
        {
            "Concordância": "Discordam (False)",
            "Quantidade": int(counts.get(False, 0)),
            "Percentual (%)": round(counts.get(False, 0) / total * 100, 2),
        },
    ]
    pd.DataFrame(rows_concorda).to_csv(
        OUT_DIR / "p5_concorda.csv", index=False, encoding="utf-8-sig"
    )
    print(f"  Salvo: p5_concorda.csv")

    # 2. Matriz de confusão
    ordem = ["POSITIVO", "TANGENCIAL", "NEGATIVO"]
    matriz = pd.crosstab(
        df["label_llm"], df["label_xgb"]
    ).reindex(index=ordem, columns=ordem, fill_value=0)
    matriz.index.name   = "LLM \\ XGBoost"
    matriz.columns.name = None
    matriz.to_csv(OUT_DIR / "p5_matriz_confusao.csv", encoding="utf-8-sig")
    print(f"  Salvo: p5_matriz_confusao.csv")
    print(matriz.to_string())

    # 3. Cohen's Kappa
    kappa = cohen_kappa_score(df["label_llm"], df["label_xgb"])
    interp = (
        "Excelente (> 0.80)"      if kappa > 0.80 else
        "Substancial (0.60–0.80)" if kappa > 0.60 else
        "Moderada (0.40–0.60)"    if kappa > 0.40 else
        "Fraca (< 0.40)"
    )
    pd.DataFrame([{"Cohen's Kappa": round(kappa, 4), "Interpretação": interp}]).to_csv(
        OUT_DIR / "p5_kappa.csv", index=False, encoding="utf-8-sig"
    )
    _stats.append({
        "Análise": "P5 — Concordância LLM vs XGBoost",
        "Teste": "Cohen's Kappa",
        "Estatística": round(kappa, 4),
        "p-valor": "—",
        "Graus de liberdade": "—",
        "Diferença significativa (α=0.05)": "—",
    })
    print(f"  Salvo: p5_kappa.csv")
    print(f"  Cohen's Kappa = {kappa:.4f} ({interp})")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("Carregando dataset...")
    df = pd.read_csv(INPUT)
    print(f"  {len(df):,} comentários carregados")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    analise_p1(df)
    analise_p2_p6(df)
    analise_p3(df)
    analise_p4(df)
    analise_p5(df)

    pd.DataFrame(_stats).to_csv(
        OUT_DIR / "estatisticas.csv", index=False, encoding="utf-8-sig"
    )
    print(f"\n  Salvo: estatisticas.csv")

    print(f"\n{'='*60}")
    print(f"Concluído! Arquivos em: {OUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
