# Modelagem e Classificação — TCC

> **Contexto:** Este documento descreve as etapas de geração de embeddings,
> treinamento e classificação do modelo, correspondendo aos scripts 05, 06 e 07.
> Registra decisões técnicas, resultados obtidos, pontos de atenção e próximos passos.

---

## Sumário

1. [Visão geral do pipeline](#1-visão-geral-do-pipeline)
2. [Script 05 — Geração de embeddings BERTimbau](#2-script-05--geração-de-embeddings-bertimbau)
3. [Script 06 — Treinamento do classificador XGBoost](#3-script-06--treinamento-do-classificador-xgboost)
4. [Script 07 — Classificação dos comentários restantes](#4-script-07--classificação-dos-comentários-restantes)
5. [O que esperamos ter após o script 07](#5-o-que-esperamos-ter-após-o-script-07)
6. [Pontos de atenção e incertezas](#6-pontos-de-atenção-e-incertezas)
7. [Próximos passos](#7-próximos-passos)

---

## 1. Visão geral do pipeline

```
preprocessed_comments.csv (54.485 comentários)
         │
         ├── 21.186 rotulados pelo LLM (llm_labeled_comments_v3.csv)
         │         │
         │         ▼ Script 05
         │   embeddings.npy (17.772 × 768)   ← exclui os 3.414 DESCARTAVEL
         │   labels.npy
         │         │
         │         ▼ Script 06
         │   XGBoost treinado + avaliado (cross-validation 5-fold)
         │   XGBoost.pkl salvo
         │
         └── 33.299 sem rótulo
                   │
                   ▼ Script 07
             BERTimbau gera embeddings
             XGBoost prediz rótulo + confiança
                   │
                   ▼
         all_labeled_comments.csv
         (54.485 comentários unificados com label + confidence)
```

---

## 2. Script 05 — Geração de embeddings BERTimbau

**Arquivo:** `scripts/05_generate_embeddings.py`

### O que faz

Transforma cada comentário da base de treinamento em um vetor numérico de
768 dimensões usando o BERTimbau. Esses vetores são a entrada do XGBoost.

### Decisões técnicas

| Decisão | Escolha | Justificativa |
|---|---|---|
| Modelo | `neuralmind/bert-base-portuguese-cased` | BERTimbau pré-treinado em português brasileiro — melhor representação semântica para o domínio |
| Coluna de texto | `text_clean` | Preserva estrutura da frase (pontuação, maiúsculas, emojis convertidos). `text_preprocessed` degradaria o BERTimbau |
| Representação | Vetor `[CLS]` da última camada | Token de início treinado para condensar o significado da frase inteira |
| `MAX_LENGTH` | 128 tokens | Cobre >99% dos comentários sem desperdício de memória |
| `BATCH_SIZE` | 32 | Equilíbrio entre velocidade e consumo de RAM |
| DESCARTAVEL | Excluído | Sem carga emocional — não deve compor o treino |

### Mapeamento de rótulos

| Label | Inteiro |
|---|---|
| NEGATIVO | 0 |
| TANGENCIAL | 1 |
| POSITIVO | 2 |
| DESCARTAVEL | excluído |

### Resultados obtidos

```
Shape dos embeddings: (17.772, 768)
```

### Arquivos gerados

| Arquivo | Descrição |
|---|---|
| `data/processed/embeddings.npy` | Matriz 17.772 × 768 de float32 |
| `data/processed/labels.npy` | Vetor 17.772 de inteiros (0/1/2) |
| `data/processed/embedding_meta.csv` | commentId, label_llm, label, channel_type, likeCount, video_title |

---

## 3. Script 06 — Treinamento do classificador XGBoost

**Arquivo:** `scripts/06_train_classifier.py`

### O que faz

Treina o XGBoost sobre os embeddings BERTimbau com rótulos conhecidos.
Avalia o modelo com validação cruzada estratificada 5-fold e salva o modelo treinado.

### Decisões técnicas

| Decisão | Escolha | Justificativa |
|---|---|---|
| Modelo | XGBoost | Recomendado pela orientadora; melhor desempenho em análise de sentimentos com embeddings |
| `sample_weight='balanced'` | Sim | NEGATIVO representa apenas 3,9% dos dados — sem balanceamento o modelo ignoraria essa classe |
| Validação | StratifiedKFold k=5 | Garante proporção de classes em cada fold; 5 folds é padrão acadêmico |
| Métricas principais | F1-macro + F1-weighted | F1-macro trata todas as classes igualmente (essencial com desbalanceamento); acurácia sozinha engana |
| `n_estimators` | 200 | Equilíbrio entre capacidade e tempo de treino |
| `max_depth` | 6 | Profundidade moderada, evita overfitting excessivo |

### Resultados obtidos (cross-validation)

| Fold | Acurácia | F1-macro | F1-weighted |
|---|---|---|---|
| 1 | 0,8560 | 0,6942 | 0,8502 |
| 2 | 0,8706 | 0,7238 | 0,8653 |
| 3 | 0,8607 | 0,6920 | 0,8547 |
| 4 | 0,8708 | 0,7267 | 0,8661 |
| 5 | 0,8692 | 0,6991 | 0,8633 |
| **Média** | **0,8655 ± 0,006** | **0,7072 ± 0,015** | **0,8599 ± 0,006** |

**Interpretação:**
- Acurácia de 86,5% sólida e consistente entre folds (desvio de apenas 0,6%)
- F1-macro de 70,7% abaixo do F1-weighted (86%) indica que o modelo tem mais
  dificuldade com NEGATIVO (classe mais rara) — esperado e documentado
- Baixo desvio padrão em todas as métricas indica boa generalização

> **Atenção:** O relatório de classificação no dataset completo (1,00 em tudo)
> é esperado — o XGBoost memorizou os dados de treino. A avaliação real é o
> cross-validation acima.

### Arquivos gerados

| Arquivo | Descrição |
|---|---|
| `data/processed/models/XGBoost.pkl` | Modelo treinado no dataset completo, pronto para inferência |
| `data/processed/models/resultados_cv.csv` | Métricas por fold |
| `data/processed/models/relatorio_treino.txt` | Relatório completo da execução |

---

## 4. Script 07 — Classificação dos comentários restantes

**Arquivo:** `scripts/07_classify_remaining.py`

### O que faz

1. Calcula **confiança** (probabilidade máxima do predict_proba) para os 17.772
   comentários LLM-labeled, usando os embeddings já existentes em `embeddings.npy`
2. Gera embeddings BERTimbau para os ~33.299 comentários sem rótulo
3. Usa o XGBoost para predizer **rótulo + confiança** para cada um
4. Une tudo e salva o dataset unificado final

### Por que salvar a confiança?

A confiança (probabilidade máxima da predição) é a métrica usada para calcular
a **intensidade emocional** no script 08, conforme definido no TCC:

| Confiança | Intensidade |
|---|---|
| > 0,80 | Alta |
| 0,60 – 0,80 | Média |
| < 0,60 | Baixa |

### Decisões técnicas

| Decisão | Escolha | Justificativa |
|---|---|---|
| Confiança LLM-labeled | Calculada via XGBoost sobre embeddings.npy | Evita re-executar BERTimbau nos 17k; reutiliza arquivo já existente |
| DESCARTAVEL | `confidence = None` | Não participam da análise de intensidade |
| Checkpoint | A cada 500 comentários | Se interrompido, retoma sem perder progresso |
| `label_source` | `"llm"` ou `"modelo"` | Rastreabilidade da origem de cada rótulo no dataset final |

### Estrutura do dataset final

```
all_labeled_comments.csv
├── commentId
├── text              ← texto original
├── text_clean        ← texto limpo (usado nos embeddings)
├── channel_type      ← "profissional" ou "amador"
├── likeCount
├── video_title
├── label             ← POSITIVO / NEGATIVO / TANGENCIAL / DESCARTAVEL
├── confidence        ← probabilidade 0.0–1.0 (None para DESCARTAVEL)
└── label_source      ← "llm" ou "modelo"
```

---

## 5. O que esperamos ter após o script 07

```
all_labeled_comments.csv com ~54.485 linhas
│
├── 21.186 com label_source = "llm"
│     ├── 8.932  POSITIVO
│     ├── 8.141  TANGENCIAL
│     ├── 699    NEGATIVO
│     └── 3.414  DESCARTAVEL (confidence = None)
│
└── 33.299 com label_source = "modelo"
      ├── POSITIVO    (proporção esperada: ~40–50%)
      ├── TANGENCIAL  (proporção esperada: ~38–45%)
      └── NEGATIVO    (proporção esperada: ~3–10%)
      ⚠️  Sem DESCARTAVEL — limitação conhecida (ver seção 6)
```

---

## 6. Pontos de atenção e incertezas

### ⚠️ Ponto crítico: DESCARTAVEL nos 33k não identificados

O XGBoost foi treinado com 3 classes (POSITIVO, NEGATIVO, TANGENCIAL).
Ele **não consegue classificar comentários como DESCARTAVEL** — qualquer
comentário que deveria ser DESCARTAVEL (spam, pergunta factual, timestamp)
vai receber um dos 3 rótulos de sentimento, contaminando a análise.

Na base LLM-labeled, 16,1% dos comentários eram DESCARTAVEL. Se essa proporção
se mantém nos 33k restantes, aproximadamente **5.400 comentários podem estar
mal classificados** nos dados de análise.

**Solução em avaliação:** rodar o script 03 nos 33k restantes para obter
rótulos LLM completos (incluindo DESCARTAVEL) — ver seção 7.

---

### ⚠️ Confiança inflada para os 17k LLM-labeled

A confiança calculada pelo XGBoost para os 17.772 comentários de treino é
computada **sobre os mesmos dados em que o modelo foi treinado**. O XGBoost
tende a ter confiança artificialmente alta nesses casos (overfitting).

**Impacto:** a análise de intensidade para os comentários LLM-labeled pode
subestimar a proporção de intensidade "baixa" nesse subconjunto.

**Mitigação possível:** usar os scores out-of-fold do cross-validation como
confiança para os 17k (mais complexo de implementar).

---

### ⚠️ Desempenho do modelo em NEGATIVO

Com apenas 699 exemplos de NEGATIVO (3,9% do treino), o modelo tem
dificuldade com essa classe mesmo com balanceamento. O F1-macro de 70,7%
(vs F1-weighted de 86%) reflete essa limitação.

**Impacto:** comentários negativos nos 33k podem ser subestimados —
alguns NEGATIVO podem ser classificados como TANGENCIAL.

---

### ℹ️ Papel do XGBoost na narrativa do TCC

**Decisão pendente:** dependendo de como os 33k forem tratados, o XGBoost
pode ter dois papéis diferentes:

| Cenário | Papel do XGBoost |
|---|---|
| 33k classificados pelo XGBoost (atual) | Classificador de produção — suas predições entram na análise |
| 33k rotulados pelo LLM (Opção A) | Modelo avaliado formalmente — comparado contra ground truth LLM |

O segundo cenário é metodologicamente mais sólido e ainda mantém o XGBoost
como componente central do pipeline, mas muda sua função no TCC.

---

## 7. Próximos passos

### Decisão imediata

Antes de criar o script 08, é necessário decidir:

**Opção A (recomendada):** Rodar o script 03 nos 33k restantes para obter
rótulos LLM completos. O script 07 é reescrito para usar LLM como ground
truth nos 33k e avaliar o XGBoost contra esses rótulos.

```bash
# Rotula os 33k restantes (retoma do checkpoint automaticamente)
python scripts/03_llm_label.py
```

**Opção B:** Aceitar as limitações do atual approach e documentá-las no TCC.
O script 07 classifica os 33k com XGBoost e a análise usa esses rótulos.

---

### Script 08 — Análise comparativa (a criar)

Independente da decisão acima, o script 08 responderá as 4 perguntas
de pesquisa usando `all_labeled_comments.csv`:

| Pergunta | Como responder |
|---|---|
| 1 — Comentários mais curtidos expressam sentimentos diferentes? | Filtrar `likeCount > 50`, comparar distribuição de labels por `channel_type` |
| 2 — Profissionais geram mais comentários positivos? | `groupby('channel_type')['label'].value_counts(normalize=True)` |
| 3 — Intensidade emocional difere entre grupos? | Comparar distribuição de `confidence` (e intensidade categórica) por `channel_type` |
| 4 — Profissionais têm mais indícios de confiabilidade percebida? | Heurística de palavras-chave: "explicou", "confio", "faz sentido" vs "errado", "não funciona" |

**Visualizações planejadas:**
- Gráficos de barras: proporção de sentimentos por grupo
- Boxplots: distribuição de confiança por grupo e label
- WordClouds: termos mais frequentes por grupo × sentimento
- Tabela resumo final: profissional vs. amador em todas as dimensões

---

*Documento gerado em 28/04/2026. Scripts em `scripts/`. Dados em `data/processed/`.*
