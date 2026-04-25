# Progresso do TCC — Análise de Impacto Emocional em Vídeos de Saúde Mental

**Universidade:** Mackenzie  
**Data de atualização:** abril de 2026

---

## Objetivo do Projeto

Identificar diferenças no impacto emocional gerado por vídeos sobre saúde mental produzidos por **profissionais** (psicólogos, psiquiatras, filósofos) versus **criadores amadores** no YouTube Brasil.

A hipótese central é que o tipo de produtor de conteúdo influencia a natureza e intensidade das reações emocionais nos comentários.

---

## Perguntas de Pesquisa

1. Os comentários mais engajados (mais likes) expressam sentimentos diferentes entre os dois grupos?
2. Conteúdos de profissionais geram maior proporção de comentários positivos?
3. A intensidade emocional dos comentários difere entre profissionais e amadores?
4. Conteúdos de profissionais apresentam mais indícios de confiabilidade percebida?

---

## Dataset

### Canais Coletados

| Canal | Tipo | Criador |
|---|---|---|
| Prazer, Karnal | Profissional | Leandro Karnal |
| Maria Homem | Profissional | Maria Homem |
| Neurologia e Psiquiatria | Profissional | Saulo Nader / Maria Fernanda Caliani |
| Rossandro Klinjey | Profissional | Rossandro Klinjey |
| PodPeople - Ana Beatriz Barbosa | Profissional | Ana Beatriz Barbosa |
| Augusto Cury | Profissional | Augusto Cury |
| Minutos Psíquicos | Profissional | — |
| Casa do Saber | Profissional | — |
| ellora | Amador | Ellora Haonne |
| JoutJout Prazer | Amador | JoutJout |
| Ludoviajante | Amador | — |
| Juliana Goes | Amador | Juliana Goes |
| Fred Elboni | Amador | Fred Elboni |
| Obvious | Amador | — |

### Números Gerais

| Métrica | Valor |
|---|---|
| Vídeos coletados | 79 |
| Comentários (após limpeza) | 42.372 |
| Profissional | 21.351 (50,4%) |
| Amador | 21.021 (49,6%) |
| Período dos comentários | nov/2019 — abr/2026 |

---

## Pipeline de Processamento

### Etapa 1 — Coleta (`scripts/00_collect.py`)

Comentários coletados via YouTube Data API v3. A coleta foi feita em duas rodadas:
- Rodada 1: dataset inicial (14.762 comentários)
- Rodada 2: expansão com novos canais (~28.000 comentários adicionais)

Os dois lotes foram mesclados e deduplicados por `commentId`, resultando em 42.372 comentários únicos.

**Arquivo gerado:** `raw_comments.csv`, `video_metadata.csv`

---

### Etapa 2 — Limpeza Estrutural (`scripts/01_clean_data.py`)

Limpeza de qualidade independente de NLP:

- Remoção de textos nulos e vazios após strip
- Deduplicação por `commentId`
- Conversão de `publishedAt` para datetime UTC
- Validação de `likeCount` não-negativo
- Criação da coluna `text_length`
- Criação da flag `is_substantive` (comentário contém ao menos uma letra)
- Inferência de `channel_type` a partir do `channelTitle`

**Arquivos gerados:** `data/processed/cleaned_comments.csv`, `data/processed/cleaned_metadata.csv`

---

### Etapa 3 — Pré-processamento de Texto (`scripts/02_preprocess_text.py`)

Duas versões de texto criadas para usos diferentes:

| Versão | Uso | Transformações |
|---|---|---|
| `text_clean` | BERTimbau (embeddings) | Remove URLs, mencões, hashtags; converte emojis para texto em português via `emoji` library |
| `text_preprocessed` | TF-IDF / análise léxica | Tudo acima + lowercase, remoção de acentos, stopwords (NLTK pt), stemming (RSLP) |

**Arquivo gerado:** `data/processed/preprocessed_comments.csv` (42.372 linhas, 11 colunas)

---

### Etapa 4 — Rotulação via LLM (`scripts/03_llm_label.py`)

Classificação automática dos comentários usando **Claude Haiku** (`claude-haiku-4-5-20251001`) via API da Anthropic.

#### Schema de labels

| Label | Descrição |
|---|---|
| **POSITIVO** | Reação emocional positiva **direcionada ao vídeo** ou ao criador (gratidão, alívio, elogio, esperança gerada pelo vídeo) |
| **NEGATIVO** | Reação emocional negativa **direcionada ao vídeo** ou ao criador (crítica ao conteúdo, discordância, denúncia de desinformação) |
| **TANGENCIAL** | Comentário que tangencia o vídeo sem ser uma reação direta a ele (relato pessoal, opinião sobre o tema/assunto, reação a eventos mencionados) |
| **DESCARTAVEL** | Sem sentimento emocional relevante (perguntas factuais, spam, timestamps, emojis ambíguos isolados) |

#### Iterações de rotulação

**v1 (descartada):** 3 labels (POSITIVO, NEGATIVO, DESCARTAVEL), sem contexto do título do vídeo. Descartada por alta taxa de falsos negativos — comentários de desabafo pessoal classificados como NEGATIVO.

**v2:** Adicionado label VIVENCIAL para capturar relatos pessoais. Identificou-se posteriormente que ~13% dos comentários NEGATIVOS eram na verdade reações ao *tema* do vídeo (ex: comentários de indignação em vídeo sobre crimes), não ao vídeo em si.

**v3 (atual):** Duas melhorias:
1. O título de cada vídeo é incluído no prompt junto ao comentário (`[Vídeo: "título"]`), dando contexto para o modelo distinguir "reação ao vídeo" de "reação ao tema"
2. Label renomeado de VIVENCIAL para **TANGENCIAL** (nomenclatura mais precisa — cobre tanto relatos pessoais quanto opiniões sobre o tema)

#### Resultados v3 (primeira metade do dataset)

| Label | Contagem | % |
|---|---|---|
| POSITIVO | 8.932 | 42,2% |
| TANGENCIAL | 8.141 | 38,4% |
| DESCARTAVEL | 3.414 | 16,1% |
| NEGATIVO | 699 | 3,3% |
| **Total rotulado** | **21.186** | — |

**Custo:** R$ 16,54 (US$ 2,85) — dentro do limite de R$ 40,00

#### Detalhes técnicos da rotulação

- Processamento em lotes de 20 comentários por chamada (economia de tokens)
- Checkpoint automático: retoma de onde parou se interrompido
- Rate limiting: pausa de 2,5s entre lotes; 65s em caso de 429
- Max retries: 4 tentativas por lote; fallback para DESCARTAVEL em parse errors

---

## Estado Atual

| Etapa | Status |
|---|---|
| Coleta de dados | Concluída |
| Limpeza estrutural | Concluída |
| Pré-processamento de texto | Concluído |
| Rotulação LLM (1ª metade) | Concluída — v3 |
| Revisão humana (600 comentários) | Pendente |
| Rotulação LLM (2ª metade) | Pendente |
| Construção do dataset final | Pendente |
| Análise e modelagem | Pendente |

---

## Próximos Passos Imediatos

1. **Gerar amostras para revisão humana** — `scripts/04_sample_for_review.py`  
   Seleciona 600 comentários estratificados → divide em `para_revisao_arthur.csv` e `para_revisao_ana.csv` (300 cada)

2. **Revisão humana** — Arthur e Ana preenchem a coluna `label_revisao`  
   Avaliar concordância com o modelo (métrica: Cohen's Kappa)

3. **Rotular a 2ª metade** — rodar `03_llm_label.py` sem `--limit`

4. **Construir dataset final** — unir labels + comentários + metadados

5. **Análise e modelagem** — embeddings BERTimbau, comparação profissional × amador

---

## Estrutura de Arquivos Relevantes

```
TCC/
├── raw_comments.csv                          # Comentários brutos (42.382 antes dedup)
├── video_metadata.csv                        # Metadados dos 79 vídeos
├── scripts/
│   ├── 01_clean_data.py                      # Limpeza estrutural
│   ├── 02_preprocess_text.py                 # Pré-processamento NLP
│   ├── 03_llm_label.py                       # Rotulação via Claude Haiku (v3)
│   └── 04_sample_for_review.py              # Amostragem para revisão humana
└── data/processed/
    ├── cleaned_comments.csv
    ├── cleaned_metadata.csv
    ├── preprocessed_comments.csv             # 42.372 linhas, texto limpo + pré-processado
    ├── llm_labeled_comments_v3.csv           # 21.186 comentários rotulados (1ª metade)
    └── llm_checkpoint_v3.json               # Checkpoint de progresso
```
