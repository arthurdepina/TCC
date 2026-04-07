"""
03_llm_label.py
---------------
Rotula comentários de saúde mental usando Claude Haiku via API da Anthropic.

Funcionalidades:
  - Processa em lotes de 20 comentários por chamada (economiza tokens)
  - Checkpoint automático: se interrompido, retoma de onde parou
  - Limite de custo configurável em R$ (para ao atingir 90% do limite)
  - Parâmetro --limit para processar apenas N comentários

Entrada:
    data/processed/preprocessed_comments.csv

Saídas:
    data/processed/llm_labeled_comments.csv   — resultado acumulado
    data/processed/llm_checkpoint.json        — progresso salvo

Uso:
    # Metade do dataset (~21k comentários):
    python scripts/03_llm_label.py --limit 21186

    # Dataset completo:
    python scripts/03_llm_label.py

    # Retomar de onde parou (mesmo comando, detecta checkpoint automático):
    python scripts/03_llm_label.py --limit 21186

Requer:
    ANTHROPIC_API_KEY no arquivo .env (na raiz do projeto)
"""

import os
import sys
import json
import time
import argparse
import io
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
import anthropic

# ─── Configurações ────────────────────────────────────────────────────────────

BATCH_SIZE    = 20                 # comentários por chamada de API
MODEL         = "claude-haiku-4-5-20251001"

# Custo Claude Haiku (USD por milhão de tokens)
INPUT_COST_PER_M  = 0.80
OUTPUT_COST_PER_M = 4.00

USD_TO_BRL    = 5.80               # taxa aproximada; ajuste se necessário
MAX_COST_BRL  = 40.0               # limite de custo em R$
WARN_THRESHOLD = 0.90              # avisa ao atingir 90% do limite

MAX_RETRIES   = 3                  # tentativas por lote em caso de erro de API
RETRY_WAIT    = 5                  # segundos entre tentativas

# ─── Caminhos ────────────────────────────────────────────────────────────────

ROOT        = Path(__file__).parent.parent
INPUT_CSV   = ROOT / "data" / "processed" / "preprocessed_comments.csv"
OUTPUT_CSV  = ROOT / "data" / "processed" / "llm_labeled_comments.csv"
CHECKPOINT  = ROOT / "data" / "processed" / "llm_checkpoint.json"

# ─── Prompt ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
Você é um especialista em análise de sentimentos aplicada a saúde mental.
Sua tarefa é classificar comentários de vídeos do YouTube sobre saúde mental
e bem-estar no Brasil.

Classifique cada comentário em UMA das três categorias:
- POSITIVO: expressa emoção positiva em relação ao conteúdo do vídeo
  (gratidão, alívio, identificação, esperança, apoio, elogio ao criador)
- NEGATIVO: expressa emoção negativa em relação ao conteúdo do vídeo
  (tristeza, angústia, crítica, discordância forte, desconforto, sofrimento)
- DESCARTAVEL: não expressa sentimento emocional claro sobre o conteúdo
  (perguntas factuais, spam, propaganda, timestamps, emojis ambíguos)

Regras:
- Sentimento misto: use o dominante ou o da última oração
- Ironia/sarcasmo: classifique pelo sentimento real, não pelo superficial
- Emoção sobre a própria situação de vida: classifique como NEGATIVO
  (o conteúdo ativou essa emoção)
- Emojis de coração isolados (💛💜💙❤): POSITIVO
- Emojis ambíguos isolados (👀🤔): DESCARTAVEL

Responda APENAS com um array JSON contendo exatamente N strings, uma por
comentário, na mesma ordem. Exemplo para 3 comentários:
["POSITIVO", "NEGATIVO", "DESCARTAVEL"]
"""

FEW_SHOT_USER = """\
Classifique estes 6 comentários:
1. Esse vídeo me fez chorar de alívio. Obrigada por falar sobre isso 💛
2. Discordo completamente. Isso é uma simplificação perigosa da depressão.
3. Qual o nome do livro que você mencionou aos 3:45?
4. Não consigo nem terminar de assistir porque me traz muitas memórias ruins
5. Finalmente alguém explicando isso de forma acessível!
6. 👀"""

FEW_SHOT_ASSISTANT = '["POSITIVO", "NEGATIVO", "DESCARTAVEL", "NEGATIVO", "POSITIVO", "DESCARTAVEL"]'

# ─── Rastreamento de custo ─────────────────────────────────────────────────────

class CostTracker:
    def __init__(self, max_brl: float):
        self.input_tokens  = 0
        self.output_tokens = 0
        self.max_brl       = max_brl

    def add(self, usage):
        self.input_tokens  += usage.input_tokens
        self.output_tokens += usage.output_tokens

    @property
    def cost_usd(self) -> float:
        return (self.input_tokens  / 1_000_000 * INPUT_COST_PER_M
              + self.output_tokens / 1_000_000 * OUTPUT_COST_PER_M)

    @property
    def cost_brl(self) -> float:
        return self.cost_usd * USD_TO_BRL

    def status(self) -> str:
        return (f"R${self.cost_brl:.2f} / R${self.max_brl:.2f}  "
                f"(US${self.cost_usd:.3f} | "
                f"{self.input_tokens:,} in + {self.output_tokens:,} out tokens)")

    def should_warn(self) -> bool:
        return self.cost_brl >= self.max_brl * WARN_THRESHOLD

    def limit_reached(self) -> bool:
        return self.cost_brl >= self.max_brl


# ─── Checkpoint ───────────────────────────────────────────────────────────────

def load_checkpoint() -> set:
    """Retorna set de commentIds já rotulados."""
    if CHECKPOINT.exists():
        with open(CHECKPOINT, "r", encoding="utf-8") as f:
            data = json.load(f)
        return set(data.get("labeled_ids", []))
    return set()


def save_checkpoint(labeled_ids: set) -> None:
    with open(CHECKPOINT, "w", encoding="utf-8") as f:
        json.dump({"labeled_ids": list(labeled_ids)}, f)


# ─── Rotulação via API ────────────────────────────────────────────────────────

def label_batch(client: anthropic.Anthropic, texts: list[str]) -> list[str]:
    """
    Envia um lote de textos para o Claude e retorna lista de rótulos.
    Tenta até MAX_RETRIES vezes em caso de falha de parsing ou erro de API.
    """
    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
    user_msg = f"Classifique estes {len(texts)} comentários:\n{numbered}"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=200,
                system=SYSTEM_PROMPT,
                messages=[
                    {"role": "user",      "content": FEW_SHOT_USER},
                    {"role": "assistant", "content": FEW_SHOT_ASSISTANT},
                    {"role": "user",      "content": user_msg},
                ],
            )
            raw = response.content[0].text.strip()

            # Extrai o array JSON da resposta
            start = raw.find("[")
            end   = raw.rfind("]") + 1
            if start == -1 or end == 0:
                raise ValueError(f"JSON nao encontrado na resposta: {raw[:100]}")

            labels = json.loads(raw[start:end])

            # Valida comprimento e valores
            valid = {"POSITIVO", "NEGATIVO", "DESCARTAVEL"}
            if len(labels) != len(texts):
                raise ValueError(
                    f"Esperado {len(texts)} labels, recebido {len(labels)}"
                )
            labels = [l.upper().strip() for l in labels]
            for l in labels:
                if l not in valid:
                    raise ValueError(f"Label invalido: '{l}'")

            return labels, response.usage

        except (anthropic.APIError, anthropic.RateLimitError) as e:
            print(f"    [API erro tentativa {attempt}/{MAX_RETRIES}]: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_WAIT * attempt)
            else:
                raise

        except (ValueError, json.JSONDecodeError) as e:
            print(f"    [Parse erro tentativa {attempt}/{MAX_RETRIES}]: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(2)
            else:
                # Fallback: retorna DESCARTAVEL para o lote problemático
                print("    Fallback: marcando lote como DESCARTAVEL")
                dummy_usage = type("U", (), {"input_tokens": 0, "output_tokens": 0})()
                return ["DESCARTAVEL"] * len(texts), dummy_usage


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Rotula comentários via Claude Haiku")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Numero maximo de comentarios a processar (ex: 21186 para metade)"
    )
    parser.add_argument(
        "--max-cost-brl", type=float, default=MAX_COST_BRL,
        help=f"Limite de custo em R$ (padrao: {MAX_COST_BRL})"
    )
    args = parser.parse_args()

    # Carrega .env
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERRO: ANTHROPIC_API_KEY nao encontrada.")
        print("Verifique se o arquivo .env existe e contém ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    client  = anthropic.Anthropic(api_key=api_key)
    tracker = CostTracker(max_brl=args.max_cost_brl)

    print("=" * 55)
    print("  Rotulacao via LLM (Claude Haiku) - TCC")
    print("=" * 55)
    print(f"  Limite de custo : R${args.max_cost_brl:.2f}")
    print(f"  Limite de linhas: {args.limit if args.limit else 'todos'}")

    # Carrega dataset
    df = pd.read_csv(INPUT_CSV)
    if args.limit:
        df = df.head(args.limit)
    total = len(df)
    print(f"  Comentarios alvo: {total:,}")

    # Checkpoint: pula já rotulados
    labeled_ids = load_checkpoint()
    df_pending = df[~df["commentId"].isin(labeled_ids)].copy()
    print(f"  Ja rotulados    : {len(labeled_ids):,}")
    print(f"  Pendentes       : {len(df_pending):,}")

    if df_pending.empty:
        print("\n  Todos os comentarios ja foram rotulados!")
        print("  Rode 04_build_dataset.py para gerar o arquivo final.")
        return

    # Carrega resultados parciais existentes
    if OUTPUT_CSV.exists():
        results_df = pd.read_csv(OUTPUT_CSV)
        results = results_df.to_dict("records")
    else:
        results = []

    # Processa em lotes
    texts  = df_pending["text_clean"].fillna("").tolist()
    ids    = df_pending["commentId"].tolist()
    n_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"\n  Iniciando em {n_batches} lotes de ate {BATCH_SIZE} comentarios...\n")

    for batch_idx in range(n_batches):
        start = batch_idx * BATCH_SIZE
        end   = min(start + BATCH_SIZE, len(texts))
        batch_texts = texts[start:end]
        batch_ids   = ids[start:end]

        # Limite de custo
        if tracker.limit_reached():
            print(f"\n  LIMITE DE CUSTO ATINGIDO: {tracker.status()}")
            print("  Progresso salvo. Rode novamente para continuar.")
            break

        if tracker.should_warn():
            print(f"  AVISO: {tracker.status()}")

        labels, usage = label_batch(client, batch_texts)
        tracker.add(usage)

        for cid, label in zip(batch_ids, labels):
            results.append({"commentId": cid, "label_llm": label})
            labeled_ids.add(cid)

        # Salva progresso a cada lote
        pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
        save_checkpoint(labeled_ids)

        # Progresso
        done = min(end + len(labeled_ids) - len(df_pending), total)
        pct  = (batch_idx + 1) / n_batches * 100
        print(f"  Lote {batch_idx+1:>4}/{n_batches} | {pct:5.1f}% | {tracker.status()}")

    print(f"\n  Rotulacao concluida.")
    print(f"  Custo final: {tracker.status()}")
    print(f"  Rotulos salvos em: {OUTPUT_CSV}")
    print(f"  Total rotulado: {len(results):,} de {total:,} comentarios")

    if len(results) < total:
        print(f"\n  {total - len(results):,} comentarios pendentes.")
        print("  Rode o script novamente para continuar (checkpoint ativo).")
    else:
        print("\n  Todos os comentarios foram rotulados!")
        print("  Proximo passo: python scripts/04_build_dataset.py")


if __name__ == "__main__":
    main()
