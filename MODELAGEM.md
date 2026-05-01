# Modelagem e Classificação — TCC

> **Contexto:** Este documento descreve as etapas de geração de embeddings,
> treinamento e construção do dataset final, correspondendo aos scripts 05, 06 e 07.
> Registra decisões técnicas, resultados obtidos e justificativas.

---

## Sumário

1. [Visão geral do pipeline](#1-visão-geral-do-pipeline)
2. [Script 05 — Geração de embeddings BERTimbau](#2-script-05--geração-de-embeddings-bertimbau)
3. [Script 06 — Treinamento do classificador XGBoost](#3-script-06--treinamento-do-classificador-xgboost)
4. [Script 07 — Construção do dataset final](#4-script-07--construção-do-dataset-final)
5. [Dataset final](#5-dataset-final)
6. [Próximos passos](#6-próximos-passos)

---

## 1. Visão geral do pipeline

```
llm_labeled_comments_v3.csv  (42.372 comentários — 100% rotulados pelo LLM)
         │
         ├── 34.491 não-DESCARTAVEL
         │         │
         │         ▼ Script 05
         │   embeddings.npy  (34.491 × 768)
         │   labels.npy
         │         │
         │         ▼ Script 06
         │   XGBoost treinado + avaliado (cross-validation 5-fold)
         │   F1-macro: 73.3%  |  Acurácia: 86.0%
         │   XGBoost.pkl salvo
         │         │
         │         ▼ Script 07
         │   Confiança calculada para os 34.491 (predict_proba)
         │
         └── 7.881 DESCARTAVEL  →  confidence = None
                   │
                   ▼
         all_labeled_comments.csv
         (42.372 comentários: label LLM + confidence XGBoost)
```

### Decisão: Opção A

O LLM rotulou **100% dos comentários** (42.372). O XGBoost foi treinado sobre
esses rótulos com cross-validation e serve como:

1. **Componente técnico validado** — F1-macro 73,3% no CV demonstra capacidade
   de generalização do pipeline BERTimbau + XGBoost
2. **Gerador de confiança** — o `predict_proba` do XGBoost é usado para
   calcular a intensidade emocional de cada comentário

O dataset final usa os **rótulos LLM** como fonte principal de análise.
A alternativa (XGBoost classificar a segunda metade) foi descartada pois o
modelo não consegue identificar DESCARTAVEL, contaminando ~16% dos dados.

---

## 2. Script 05 — Geração de embeddings BERTimbau

**Arquivo:** `scripts/05_generate_embeddings.py`

### O que faz

Transforma cada comentário não-DESCARTAVEL em um vetor de 768 dimensões
usando o BERTimbau. Esses vetores são a entrada do XGBoost.

### Decisões técnicas

| Decisão | Escolha | Justificativa |
|---|---|---|
| Modelo | `neuralmind/bert-base-portuguese-cased` | BERTimbau pré-treinado em português brasileiro |
| Coluna de texto | `text_clean` | Preserva pontuação, maiúsculas e emojis convertidos — importante para o BERT |
| Representação | Vetor `[CLS]` da última camada | Token treinado para condensar o significado da frase inteira |
| `MAX_LENGTH` | 128 tokens | Cobre >99% dos comentários |
| `BATCH_SIZE` | 32 | Equilíbrio entre velocidade e VRAM |
| Dispositivo | GPU (RTX 4060 Laptop, 8GB) | ~8 it/s vs ~0,3 it/s na CPU — 2 min vs ~60 min |
| DESCARTAVEL | Excluído | Sem carga emocional — não compõe o treino |

### Mapeamento de rótulos

| Label | Inteiro |
|---|---|
| NEGATIVO | 0 |
| TANGENCIAL | 1 |
| POSITIVO | 2 |
| DESCARTAVEL | excluído |

### Resultados

```
Comentários para embedding : 34.491
  NEGATIVO    (0):  1.783  ( 5,2%)
  TANGENCIAL  (1): 14.455  (41,9%)
  POSITIVO    (2): 18.253  (52,9%)

Shape dos embeddings: (34.491, 768)
Tempo de execução   : ~2 minutos (GPU RTX 4060 Laptop)
```

### Arquivos gerados

| Arquivo | Descrição |
|---|---|
| `data/processed/embeddings.npy` | Matriz 34.491 × 768 de float32 |
| `data/processed/labels.npy` | Vetor 34.491 de inteiros (0/1/2) |
| `data/processed/embedding_meta.csv` | commentId, label_llm, label, channel_type, likeCount, video_title |

---

## 3. Script 06 — Treinamento do classificador XGBoost

**Arquivo:** `scripts/06_train_classifier.py`

### O que faz

Treina o XGBoost sobre os embeddings BERTimbau com validação cruzada
estratificada 5-fold e salva o modelo treinado.

### Decisões técnicas

| Decisão | Escolha | Justificativa |
|---|---|---|
| Modelo | XGBoost | Recomendado pela orientadora; bom desempenho com embeddings densos |
| `sample_weight='balanced'` | Sim | NEGATIVO = 5,2% dos dados — sem balanceamento o modelo ignoraria essa classe |
| Validação | StratifiedKFold k=5 | Garante proporção de classes em cada fold; padrão acadêmico |
| Métricas principais | F1-macro + F1-weighted | F1-macro trata todas as classes igualmente — essencial com desbalanceamento |
| `n_estimators` | 200 | Equilíbrio entre capacidade e tempo de treino |
| `max_depth` | 6 | Profundidade moderada; evita overfitting |
| `learning_rate` | 0,1 | Padrão estável para esse número de estimadores |

### Resultados — cross-validation 5-fold

| Fold | Acurácia | F1-macro | F1-weighted |
|---|---|---|---|
| 1 | 0,8545 | 0,7254 | 0,8522 |
| 2 | 0,8636 | 0,7382 | 0,8609 |
| 3 | 0,8653 | 0,7346 | 0,8623 |
| 4 | 0,8620 | 0,7334 | 0,8598 |
| 5 | 0,8545 | 0,7314 | 0,8530 |
| **Média** | **0,8600 ± 0,005** | **0,7326 ± 0,004** | **0,8576 ± 0,004** |

**Comparação com execução anterior (17.772 amostras):**

| Métrica | v1 (17k) | v2 (34k) | Variação |
|---|---|---|---|
| Acurácia | 0,8655 ± 0,006 | 0,8600 ± 0,005 | Estável |
| F1-macro | 0,7072 ± 0,015 | **0,7326 ± 0,004** | +2,5pp, desvio -3x |
| F1-weighted | 0,8599 ± 0,006 | 0,8576 ± 0,004 | Estável |

O F1-macro melhorou com o dobro de dados, especialmente por NEGATIVO passar de
699 para 1.783 exemplos. O desvio padrão caiu pela metade — o modelo generaliza
de forma mais consistente.

**Interpretação:**
- Acurácia 86% sólida e consistente (desvio de 0,5%)
- F1-macro 73,3% abaixo do F1-weighted (85,8%) indica dificuldade residual
  com NEGATIVO (5,2% dos dados) — esperado e documentado
- Hiperparâmetros avaliados como adequados — tuning não necessário

### Arquivos gerados

| Arquivo | Descrição |
|---|---|
| `data/processed/models/XGBoost.pkl` | Modelo treinado no dataset completo |
| `data/processed/models/resultados_cv.csv` | Métricas por fold |
| `data/processed/models/relatorio_treino.txt` | Relatório completo da execução |

---

## 4. Script 07 — Construção do dataset final

**Arquivo:** `scripts/07_build_final_dataset.py`

> O script original `07_classify_remaining.py` foi concebido para a Opção B
> (XGBoost classifica a segunda metade). Com a adoção da Opção A (LLM rotula tudo),
> foi substituído por este script de lógica simplificada.

### O que faz

1. Carrega `embeddings.npy` e calcula confiança (`predict_proba`) para os
   34.491 não-DESCARTAVEL — sem precisar rodar o BERTimbau novamente
2. Reporta concordância XGBoost × LLM (informativa — XGBoost viu esses dados)
3. Monta `all_labeled_comments.csv` com labels LLM + confiança XGBoost

### Concordância XGBoost × LLM (base de treino)

> **Atenção:** O XGBoost foi treinado sobre esses dados — use o cross-validation
> do script 06 como avaliação real. Esses números refletem memorização parcial.

```
              precision    recall  f1-score   support
    NEGATIVO       0.96      1.00      0.98      1.783
  TANGENCIAL       0.98      0.98      0.98     14.455
    POSITIVO       0.99      0.98      0.99     18.253
    accuracy                           0.98     34.491
```

### Intensidade emocional

A confiança do XGBoost é usada como proxy de intensidade emocional:

| Faixa | Intensidade | Contagem | % |
|---|---|---|---|
| > 0,80 | ALTA | 29.866 | 86,6% |
| 0,60 – 0,80 | MÉDIA | 3.565 | 10,3% |
| < 0,60 | BAIXA | 1.060 | 3,1% |

Confiança média por label:

| Label | Confiança média |
|---|---|
| NEGATIVO | 0,943 |
| POSITIVO | 0,928 |
| TANGENCIAL | 0,896 |

NEGATIVO tem confiança mais alta — comentários negativos possuem características
linguísticas mais distintas, tornando-os mais fáceis de identificar pelo modelo.

---

## 5. Dataset final

**Arquivo:** `data/processed/all_labeled_comments.csv`

### Estrutura

| Coluna | Descrição |
|---|---|
| `commentId` | ID único do comentário |
| `text` | Texto original |
| `text_clean` | Texto limpo (URLs removidas, emojis convertidos) |
| `channel_type` | `"profissional"` ou `"amador"` |
| `likeCount` | Número de curtidas |
| `video_title` | Título do vídeo |
| `label` | Rótulo LLM: POSITIVO / NEGATIVO / TANGENCIAL / DESCARTAVEL |
| `confidence` | Confiança XGBoost (0,0–1,0); `None` para DESCARTAVEL |

### Distribuição

| Label | Contagem | % |
|---|---|---|
| POSITIVO | 18.253 | 43,1% |
| TANGENCIAL | 14.455 | 34,1% |
| DESCARTAVEL | 7.881 | 18,6% |
| NEGATIVO | 1.783 | 4,2% |
| **Total** | **42.372** | — |

| Canal | Contagem | % |
|---|---|---|
| Profissional | 21.351 | 50,4% |
| Amador | 21.021 | 49,6% |

---

## 6. Próximos passos

### Script 08 — Análise comparativa (a criar)

Responde as 4 perguntas de pesquisa usando `all_labeled_comments.csv`:

| Pergunta | Como responder |
|---|---|
| 1 — Comentários mais curtidos expressam sentimentos diferentes? | Filtrar `likeCount > 50`, comparar distribuição de labels por `channel_type` |
| 2 — Profissionais geram mais comentários positivos? | `groupby('channel_type')['label'].value_counts(normalize=True)` |
| 3 — Intensidade emocional difere entre grupos? | Comparar distribuição de `confidence` por `channel_type` e `label` |
| 4 — Profissionais têm mais indícios de confiabilidade percebida? | Heurística de palavras-chave: "explicou", "confio", "faz sentido" etc. |

**Visualizações planejadas:**
- Gráficos de barras: proporção de sentimentos por grupo
- Boxplots: distribuição de confiança por grupo e label
- WordClouds: termos mais frequentes por grupo × sentimento
- Tabela resumo: profissional vs. amador em todas as dimensões

### Revisão humana (pendente)

Os 600 comentários amostrados (`para_revisao_arthur.csv` e `para_revisao_ana.csv`)
devem ser revisados manualmente. Após a revisão, calcular Cohen's Kappa entre
LLM e revisores para validar a qualidade da rotulação.

---

*Documento atualizado em 01/05/2026. Scripts em `scripts/`. Dados em `data/processed/`.*
