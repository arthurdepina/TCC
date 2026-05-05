# Modelagem e Classificação — TCC

> **Contexto:** Este documento descreve as etapas de geração de embeddings,
> treinamento, construção do dataset final e extração de confiança percebida,
> correspondendo aos scripts 05, 06, 06b, 07 e 08.
> Registra decisões técnicas, resultados obtidos e justificativas.

---

## Sumário

1. [Visão geral do pipeline](#1-visão-geral-do-pipeline)
2. [Métricas de avaliação](#2-métricas-de-avaliação)
3. [Script 05 — Geração de embeddings BERTimbau](#3-script-05--geração-de-embeddings-bertimbau)
4. [Script 06 — Validação Cruzada do Pipeline](#4-script-06--validação-cruzada-do-pipeline)
5. [Script 06b — Treinamento e Predição (Half-Split)](#5-script-06b--treinamento-e-predição-half-split)
6. [Script 07 — Construção do dataset rotulado](#6-script-07--construção-do-dataset-rotulado)
7. [Script 08 — Confiança Percebida (heurística)](#7-script-08--confiança-percebida-heurística)
8. [Dataset final](#8-dataset-final)
9. [Script 09 — Análise de Resultados](#9-script-09--análise-de-resultados)
10. [Próximos passos](#10-próximos-passos)

---

## 1. Visão geral do pipeline

```
llm_labeled_comments_v3.csv  (42.372 comentários — 100% rotulados pelo LLM)
         │
         ├── 34.491 não-DESCARTAVEL
         │         │
         │         ▼ Script 05
         │   embeddings.npy  (34.491 × 768)  /  labels.npy
         │         │
         │    ┌────┴────┐
         │    │         │
         │    ▼         ▼
         │  Script 06             Script 06b
         │  cv_validation         train_and_predict
         │  CV 5-fold             Treina na 1ª metade (17.772)
         │  F1-macro: 73.3%       Prediz a 2ª metade (16.719)
         │  Acurácia: 86.0%       XGBoost_half.pkl salvo
         │  (validação acadêmica) second_half_predictions.csv
         │    │
         │    └─────────────────────┐
         │                         ▼ Script 07
         │               Join com preprocessed_comments.csv
         │               + cálculo de keywords
         │
         └── 7.881 DESCARTAVEL  →  excluídos do dataset final
                                            │
                                            ▼
                                   final_dataset.csv
                         (16.719 comentários — apenas 2ª metade:
                          label LLM + label XGBoost + confidence + keywords)
```

### Decisão: pipeline em duas etapas

O LLM rotulou **100% dos comentários** (42.372) — incluindo os 7.881 DESCARTAVEL
que o XGBoost não conseguiria identificar.

O pipeline separa claramente dois papéis:

1. **Script 06 — validação acadêmica:** CV 5-fold sobre os 34.491 não-DESCARTAVEL
   demonstra que o pipeline BERTimbau + XGBoost generaliza (F1-macro 73,3%).
   Nenhum modelo é salvo — essa etapa existe apenas para citar no TCC.

2. **Script 06b — classificação genuína:** XGBoost treina na 1ª metade (17.772)
   e prediz a 2ª metade (16.719) sem ter visto esses dados. O `confidence`
   produzido aqui é real e é usado como proxy de intensidade emocional.

O **dataset final** (`final_dataset.csv`) contém exclusivamente os 16.719
comentários da 2ª metade — com rótulo LLM (referência) e rótulo XGBoost
(predição genuína) para comparação.

---

## 2. Métricas de avaliação

Antes de interpretar os resultados, é importante entender o que cada métrica mede.
Todas derivam de quatro contagens básicas para cada classe:

| Termo | Significado |
|---|---|
| **VP** (Verdadeiro Positivo) | O modelo previu X e o rótulo real é X |
| **FP** (Falso Positivo) | O modelo previu X mas o rótulo real é outro |
| **FN** (Falso Negativo) | O rótulo real é X mas o modelo previu outro |

### Precision (Precisão)

> "Dos comentários que o modelo classificou como X, quantos realmente são X?"

```
Precision = VP / (VP + FP)
```

Alta precision significa poucos alarmes falsos. Uma precision de 0,93 para POSITIVO
significa que 93% dos comentários rotulados como POSITIVO pelo modelo realmente são positivos.

### Recall (Revocação ou Sensibilidade)

> "Dos comentários que realmente são X, quantos o modelo encontrou?"

```
Recall = VP / (VP + FN)
```

Alto recall significa que o modelo não deixa passar muitos casos reais.
Um recall de 0,24 para NEGATIVO significa que o modelo identifica apenas 24% dos
comentários realmente negativos — deixa 76% escaparem.

### F1-score

> Média harmônica entre Precision e Recall. Penaliza desequilíbrios entre os dois.

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

Útil quando há desbalanceamento de classes. Um F1 de 0,32 para NEGATIVO indica
que mesmo que o modelo acerte quando prevê NEGATIVO (precision razoável),
ele deixa passar a maioria dos casos reais (recall baixo).

### F1-macro vs F1-weighted

| Variante | Como calcula | Quando usar |
|---|---|---|
| **F1-macro** | Média simples do F1 de cada classe | Quando todas as classes têm igual importância analítica |
| **F1-weighted** | Média ponderada pelo número de exemplos de cada classe | Quando classes maiores importam mais |

Com desbalanceamento (NEGATIVO = 4,2%), o F1-macro é mais honesto: ele penaliza
o modelo por errar em classes raras mesmo que a acurácia geral seja alta.

### Matriz de confusão

Tabela que mostra onde o modelo erra: **linhas = rótulo real (LLM), colunas = rótulo previsto (modelo)**.
A diagonal principal são os acertos; fora da diagonal são os erros.

---

## 3. Script 05 — Geração de embeddings BERTimbau

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

## 4. Script 06 — Validação Cruzada do Pipeline

**Arquivo:** `scripts/06_cv_validation.py`

### O que faz

Avalia o pipeline BERTimbau + XGBoost via validação cruzada estratificada
5-fold sobre os 34.491 não-DESCARTAVEL. Objetivo exclusivamente acadêmico:
demonstrar a capacidade de generalização do pipeline. Nenhum modelo é salvo.

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
| `data/processed/models/resultados_cv.csv` | Métricas por fold |
| `data/processed/models/relatorio_cv.txt` | Relatório completo da execução |

---

## 5. Script 06b — Treinamento e Predição (Half-Split)

**Arquivo:** `scripts/06b_half_split_evaluation.py`

### Motivação

O cross-validation do script 06 avalia o modelo sobre os mesmos dados em que
foi treinado (em folds). Uma avaliação mais realista é treinar em um conjunto
e testar em outro completamente separado. Como o LLM rotulou 100% da base,
isso é possível respeitando a ordem original de rotulação:

- **Treino:** primeiros 21.186 comentários do `preprocessed_comments.csv`
- **Teste:** últimos 21.186 comentários (rótulos LLM disponíveis como ground truth)

O script reutiliza os embeddings já gerados — sem rodar o BERTimbau novamente.

### O que faz

1. Divide `embeddings.npy` em treino (1ª metade) e teste (2ª metade) por `commentId`
2. Treina um novo XGBoost (`XGBoost_half.pkl`) apenas na 1ª metade
3. Prediz rótulo + confiança para cada comentário da 2ª metade
4. Compara previsões XGBoost com rótulos LLM — avaliação genuína
5. Salva `second_half_predictions.csv` com label, confiança e intensidade

### Divisão dos dados (não-DESCARTAVEL)

| Conjunto | Comentários | NEGATIVO | TANGENCIAL | POSITIVO |
|---|---|---|---|---|
| Treino (1ª metade) | 17.772 | 699 (3,9%) | 8.141 (45,8%) | 8.932 (50,3%) |
| Teste (2ª metade) | 16.719 | 1.084 (6,5%) | 6.314 (37,8%) | 9.321 (55,8%) |

### Resultados — avaliação genuína (XGBoost vs LLM na 2ª metade)

```
              precision    recall  f1-score   support

    NEGATIVO       0.51      0.24      0.32      1.084
  TANGENCIAL       0.77      0.89      0.82      6.314
    POSITIVO       0.93      0.89      0.91      9.321

    accuracy                           0.85     16.719
   macro avg       0.74      0.67      0.69     16.719
weighted avg       0.84      0.85      0.84     16.719
```

**Concordância geral: 14.178 / 16.719 comentários (84,8%)**

Matriz de confusão (linhas = LLM, colunas = XGBoost):

```
             NEGATIVO  TANGENCIAL    POSITIVO
   NEGATIVO       255         739          90
  TANGENCIAL      188        5603         523
   POSITIVO        59         942        8320
```

### Intensidade emocional na 2ª metade (label XGBoost)

| Intensidade | Contagem | % |
|---|---|---|
| ALTA (> 0,80) | 13.039 | 78,0% |
| MÉDIA (0,60–0,80) | 2.437 | 14,6% |
| BAIXA (< 0,60) | 1.243 | 7,4% |

Confiança média por label XGBoost:

| Label | Confiança média |
|---|---|
| POSITIVO | 0,917 |
| TANGENCIAL | 0,867 |
| NEGATIVO | 0,688 |

### Conclusões

**O modelo generaliza bem para POSITIVO e TANGENCIAL, mas tem dificuldade séria com NEGATIVO.**

- **POSITIVO** (F1 = 0,91): classificação excelente. Comentários positivos têm
  vocabulário característico (gratidão, alívio, elogio) que o modelo aprende bem.

- **TANGENCIAL** (F1 = 0,82): bom desempenho. O modelo captura relatos pessoais
  e opiniões sobre o tema com consistência razoável.

- **NEGATIVO** (F1 = 0,32): desempenho fraco. O recall de 0,24 significa que
  o modelo deixa passar 76% dos comentários negativos reais, classificando-os
  majoritariamente como TANGENCIAL (739 de 1.084 casos).

**Por que NEGATIVO é tão difícil?**

Dois fatores combinados:

1. **Raridade:** NEGATIVO representa apenas 3,9% do treino (699 exemplos).
   Mesmo com `sample_weight='balanced'`, é difícil aprender padrões de uma
   classe com tão poucos exemplos.

2. **Proximidade linguística com TANGENCIAL:** Ambas as classes podem conter
   linguagem de desconforto ou crítica. A diferença (reação ao vídeo vs.
   reação ao tema) é sutil e requer contexto que o modelo não captura
   completamente via embeddings CLS isolados.

**Comparação entre avaliações:**

| Avaliação | Acurácia | F1-macro | Observação |
|---|---|---|---|
| CV 5-fold (script 06) | 86,0% | 73,3% | Treino e teste com overlap |
| Half-split (script 06b) | 84,8% | 69,0% | Avaliação genuína — sem overlap |

A queda de 4pp no F1-macro entre CV e half-split é esperada e indica que o
modelo tem alguma memorização, mas generaliza razoavelmente bem.

### Arquivos gerados

| Arquivo | Descrição |
|---|---|
| `data/processed/models/XGBoost_half.pkl` | Modelo treinado apenas na 1ª metade |
| `data/processed/models/resultados_half_split.txt` | Relatório completo |
| `data/processed/second_half_predictions.csv` | label_llm, label_xgb, confidence, intensidade, concorda |

---

## 6. Script 07 — Construção do dataset rotulado

**Arquivo:** `scripts/07_build_final_dataset.py`

### O que faz

1. Carrega `second_half_predictions.csv` (saída do script 06b) — 16.719 comentários
   com rótulo LLM, rótulo XGBoost, confiança e intensidade
2. Cruza com `preprocessed_comments.csv` para obter `text`, `text_clean`,
   `channel_type`, `likeCount` e `video_title`
3. Calcula coluna `keywords` — termos de confiabilidade percebida encontrados
   em `text_clean` (separados por ";")
4. Salva `final_dataset.csv` pronto para análise

### Termos de confiabilidade (heurística)

Usados para preencher a coluna `keywords`:

```
"explicou", "explicação", "faz sentido", "confio", "confiança",
"científico", "embasado", "comprovado", "especialista", "referência",
"pesquisa", "estudo", "dados", "evidência", "profissional",
"recomendo", "recomenda", "seguro", "correto", "verdade"
```

### Arquivos gerados

| Arquivo | Descrição |
|---|---|
| `data/processed/final_dataset.csv` | Dataset final — 16.719 comentários da 2ª metade |

---

## 7. Script 08 — Confiança Percebida (heurística)

**Arquivo:** `scripts/08_credibility_heuristic.py`

### O que faz

Extrai a métrica `confianca_percebida` para cada comentário da base final,
aplicando heurística de palavras-chave com tratamento de negação e
desambiguação via `label_xgb`. Aplica-se às três classes de sentimento:
POSITIVO, TANGENCIAL e NEGATIVO.

### Lógica de classificação

```
Para cada comentário:
  1. Busca termos de ALTA credibilidade em text_clean
  2. Verifica negação (janela de 3 palavras antes do termo)
     → termo negado → tratado como sinal de BAIXA
  3. Busca termos de BAIXA credibilidade
  4. Determina keyword_signal:
       só ALTA encontrada → ALTA
       só BAIXA encontrada → BAIXA
       ambas encontradas → CONFLITO
       nenhuma → NEUTRA
  5. Resolve CONFLITO via label_xgb:
       POSITIVO   → ALTA
       NEGATIVO   → BAIXA
       TANGENCIAL → NEUTRA
```

### Termos utilizados

**ALTA credibilidade:**
```
"explicou", "explicação", "faz sentido", "confio", "confiança",
"científico", "embasado", "comprovado", "especialista", "referência",
"pesquisa", "estudo", "evidência", "recomendo", "recomenda",
"correto", "verdade", "confiável", "fundamentado", "comprovou",
"funciona", "ajudou", "esclareceu", "ficou claro", "bem explicado",
"informação clara", "informação correta"
```

**BAIXA credibilidade:**
```
"errado", "equivocado", "pseudociência", "desinformação", "charlatão",
"sem base", "enganação", "falso", "mentira", "perigoso", "irresponsável",
"absurdo", "ridículo", "não funciona", "não confio", "não ajuda",
"não faz sentido", "não é verdade", "não é correto", "lixo", "besteira"
```

### Decisões técnicas

| Decisão | Escolha | Justificativa |
|---|---|---|
| Classes incluídas | POSITIVO, TANGENCIAL, NEGATIVO | Todas as três contêm sinais de credibilidade |
| Negação | Janela de 3 palavras antes do termo | Cobre a maioria dos casos em português sem NLP avançado |
| Conflito | Desambiguado via `label_xgb` | O sentimento já classificado contextualiza o sinal ambíguo |
| Limitação | Heurística insensível a contexto | Resultados indicativos — documentar no TCC |

### Arquivos gerados

| Arquivo | Descrição |
|---|---|
| `data/processed/final_dataset.csv` | Base completa + `confianca_percebida` — dataset de análise |
| `data/processed/final_dataset_audit.csv` | + `keyword_signal`, `keywords_alta`, `keywords_baixa`, `conflito` |

---

## 8. Dataset final

**Arquivo:** `data/processed/final_dataset.csv` (gerado pelo script 08)

> O arquivo intermediário `final_labeled_dataset.csv` (gerado pelo script 07)
> contém os mesmos dados sem a coluna `confianca_percebida`.

Contém exclusivamente os **16.719 comentários da 2ª metade** — aqueles que o
XGBoost nunca viu durante o treinamento. Os rótulos XGBoost são predições
genuínas, não memorização.

### Estrutura

| Coluna | Descrição |
|---|---|
| `commentId` | ID único do comentário |
| `text` | Texto original |
| `text_clean` | Texto tratado (URLs removidas, emojis convertidos) |
| `channel_type` | `"profissional"` ou `"amador"` |
| `likeCount` | Número de curtidas |
| `video_title` | Título do vídeo |
| `label_llm` | Rótulo atribuído pelo LLM (referência) |
| `label_xgb` | Rótulo predito pelo XGBoost (predição genuína) |
| `confidence` | Probabilidade máxima do XGBoost (0,0–1,0) |
| `intensidade` | ALTA (>0,80) / MEDIA (0,60–0,80) / BAIXA (<0,60) |
| `concorda` | `True` se label_llm == label_xgb |
| `keywords` | Termos de confiabilidade encontrados em text_clean — lista básica do script 07 |
| `confianca_percebida` | ALTA / BAIXA / NEUTRA — heurística com negação e desambiguação (script 08) |

### Distribuição (label_xgb — predição genuína do XGBoost)

| Label | Contagem | % |
|---|---|---|
| POSITIVO | 9.321 | 55,8% |
| TANGENCIAL | 6.314 | 37,8% |
| NEGATIVO | 1.084 | 6,5% |
| **Total** | **16.719** | — |

> Nota: DESCARTAVEL não consta pois o XGBoost foi treinado apenas nas 3 classes
> de sentimento. Os DESCARTAVEL da 2ª metade foram excluídos pelo script 05
> (não geraram embeddings) e portanto não aparecem no dataset final.

---

## 9. Script 09 — Análise de Resultados

**Arquivo:** `scripts/09_analyze_results.py`

### O que faz

Lê `final_dataset.csv` e responde as 6 perguntas de pesquisa do TCC, gerando
um arquivo CSV por análise em `data/processed/results/`. Cada arquivo contém
duas visões — percentual e valor absoluto — identificadas pela coluna `visao`.
Um arquivo consolidado (`estatisticas.csv`) reúne todos os testes estatísticos.

### Perguntas respondidas

| # | Pergunta | Filtro aplicado | Coluna analisada |
|---|---|---|---|
| P1 | Comentários mais engajados expressam sentimentos diferentes entre os grupos? | `likeCount >= 10` | `label_xgb` |
| P2 | Conteúdos de profissionais geram maior proporção de comentários positivos? | base completa | `label_xgb` |
| P3 | A intensidade emocional difere entre profissionais e amadores? | `label_xgb` IN (POSITIVO, NEGATIVO) | `intensidade` |
| P4 | Conteúdos de profissionais apresentam mais indícios de confiabilidade percebida? | `confianca_percebida` IN (ALTA, BAIXA) | `confianca_percebida` |
| P5 | Como se compara a categorização da LLM com a do XGBoost? | base completa | `label_llm` × `label_xgb` |
| P6 | Conteúdos de profissionais geram mais relatos pessoais (TANGENCIAL)? | base completa | `label_xgb` |

> P2 e P6 compartilham o mesmo CSV pois usam a mesma base e mesma estrutura de tabela.

### Decisões técnicas

| Decisão | Escolha | Justificativa |
|---|---|---|
| Testes estatísticos | Chi-quadrado (P1–P4, P6) e Cohen's Kappa (P5) | Chi² compara proporções entre grupos; Kappa mede concordância corrigida por acaso |
| Denominador dos percentuais | Total do grupo no contexto do filtro aplicado | Garante que percentuais somem 100% dentro de cada grupo |
| Filtro P3 | Exclui TANGENCIAL | Intensidade emocional só é interpretável para sentimentos diretos (positivo/negativo) |
| Filtro P4 | Exclui NEUTRA | A análise foca nos comentários com sinal claro de credibilidade — NEUTRA não discrimina |
| Threshold P1 | `likeCount >= 10` | Equilibra tamanho de amostra (323 profissional / 356 amador) e representatividade de engajamento |
| P3 — estrutura | 4 tabelas (profissional × amador × percentual × absoluto) | Cada grupo tem seu próprio arquivo; colunas: Total / Positivo / Negativo |

### Arquivos gerados

| Arquivo | Conteúdo |
|---|---|
| `p1_engajamento.csv` | Sentimentos (POSITIVO / NEGATIVO / TANGENCIAL) em comentários com likeCount ≥ 10 |
| `p2_p6_sentimentos_base.csv` | Distribuição de todos os rótulos na base completa por grupo |
| `p3_intensidade_profissional.csv` | Intensidade emocional — canais profissionais (Total / Positivo / Negativo) |
| `p3_intensidade_amador.csv` | Intensidade emocional — canais amadores (Total / Positivo / Negativo) |
| `p4_confianca.csv` | Confiança percebida ALTA e BAIXA por grupo |
| `p5_concorda.csv` | Concordância geral LLM vs XGBoost (True / False) |
| `p5_matriz_confusao.csv` | Matriz de confusão 3×3 (LLM nas linhas, XGBoost nas colunas) |
| `p5_kappa.csv` | Cohen's Kappa e interpretação |
| `estatisticas.csv` | Todos os testes estatísticos consolidados |

### Resultados obtidos

#### P1 — Comentários engajados (likeCount ≥ 10)

| Rótulo | Profissional | Amador |
|---|---|---|
| Positivo | 34,67% (112) | 36,52% (130) |
| Negativo | 5,57% (18) | 3,65% (13) |
| Tangencial | 59,75% (193) | 59,83% (213) |

**Chi² = 1,53, p = 0,4653 — Não significativa.** A distribuição de sentimentos
em comentários engajados é estatisticamente semelhante entre os dois grupos.

---

#### P2 — Proporção de positivos na base completa

| Rótulo | Profissional | Amador |
|---|---|---|
| Positivo | 33,43% (2.296) | 67,37% (6.637) |
| Negativo | 4,86% (334) | 1,71% (168) |
| Tangencial | 61,71% (4.238) | 30,92% (3.046) |

**Chi² = 1.887,33, p ≈ 0 — Significativa.** Canais amadores geram proporção
muito maior de comentários positivos; canais profissionais concentram
comentários tangenciais (relatos pessoais e referências ao tema).

---

#### P3 — Intensidade emocional

**Canais profissionais** (n=2.630 | pos=2.296 | neg=334):

| Intensidade | Total | Positivo | Negativo |
|---|---|---|---|
| Alta | 73,95% | 80,62% | 28,14% |
| Média | 16,08% | 12,85% | 38,32% |
| Baixa | 9,96% | 6,53% | 33,53% |

**Canais amadores** (n=6.805 | pos=6.637 | neg=168):

| Intensidade | Total | Positivo | Negativo |
|---|---|---|---|
| Alta | 84,10% | 85,61% | 24,40% |
| Média | 10,29% | 9,45% | 43,45% |
| Baixa | 5,61% | 4,94% | 32,14% |

**Chi² = 130,13, p ≈ 0 — Significativa.** Canais amadores têm maior proporção
de comentários de alta intensidade. Em ambos os grupos, comentários negativos
distribuem-se de forma mais equilibrada entre as três faixas de intensidade.

---

#### P4 — Confiança percebida (ALTA / BAIXA)

| Confiança | Profissional | Amador |
|---|---|---|
| Alta | 62,24% (455) | 75,69% (439) |
| Baixa | 37,76% (276) | 24,31% (141) |

**Chi² = 26,34, p ≈ 0 — Significativa.** Canais amadores apresentam maior
proporção de comentários com sinais de alta confiança percebida. Nota: 92%
da base é NEUTRA — os percentuais acima se referem apenas aos 1.311
comentários com sinal explícito de credibilidade.

---

#### P5 — Concordância LLM vs XGBoost

**Concordância geral:** 14.178 / 16.719 comentários **(84,8%)**

**Matriz de confusão** (linhas = LLM, colunas = XGBoost):

|  | POSITIVO | TANGENCIAL | NEGATIVO |
|---|---|---|---|
| **POSITIVO** | 8.320 | 942 | 59 |
| **TANGENCIAL** | 523 | 5.603 | 188 |
| **NEGATIVO** | 90 | 739 | 255 |

Principal padrão de discordância: XGBoost classifica como TANGENCIAL
739 dos 1.084 comentários que a LLM rotulou como NEGATIVO (recall de 24%
para NEGATIVO — já documentado no script 06b).

**Cohen's Kappa = 0,7163 — Concordância Substancial (0,60–0,80).**
A concordância real supera substancialmente o que seria esperado por acaso,
validando o uso do XGBoost como classificador complementar à LLM.

---

#### P6 — Comentários TANGENCIAIS por grupo

Coberto pelo mesmo CSV de P2. Profissional: 61,71% (4.238) vs Amador: 30,92% (3.046).
Diferença altamente significativa — discutida junto a P2.

---

## 10. Próximos passos

A análise quantitativa está concluída. Os resultados em `data/processed/results/`
estão prontos para uso na escrita do TCC.

---

*Documento atualizado em 02/05/2026. Scripts em [scripts/](scripts/). Dados em [data/processed/](data/processed/).*
