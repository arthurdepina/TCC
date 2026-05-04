# Handoff — Continuação do Relatório (TCC Parte 2)

Este documento é um guia para a coautora continuar a escrita do artigo no Claude.
Ele resume tudo o que já foi feito, o que ainda falta, os dados reais para usar e as regras a seguir.

---

## 1. Contexto do projeto

**Título:** Análise comparativa do impacto de conteúdos sobre saúde mental no YouTube por meio de comentários: profissionais vs. não profissionais

**Autores:** Arthur de Pina Balduino Leitão e Carolina Toledo Ferraz
**Instituição:** Faculdade de Computação e Informática (FCI), Universidade Presbiteriana Mackenzie

**O que é o projeto:**
Análise de sentimentos em comentários do YouTube de 12 canais de saúde mental e bem-estar — seis operados por profissionais de saúde (médicos, psicólogos) e seis por criadores amadores (sem formação técnica). O objetivo é comparar o impacto emocional entre os dois grupos por meio de seis perguntas de pesquisa.

**Arquivos principais:**
- `artigo.md` — o artigo em Markdown (editar aqui)
- `referencias.bib` — todas as referências em BibTeX
- `build.ps1` — script que compila o artigo para `artigo_final.docx`

**Como compilar:**
```powershell
# No terminal, dentro da pasta relatorio2/:
.\build.ps1
```
O arquivo `artigo_final.docx` abre automaticamente com a formatação da Mackenzie.

**Como citar no texto:**
- `[@autor2020]` → (AUTOR, 2020) — citação entre parênteses
- `@autor2020` → Autor (2020) — autor faz parte da frase

---

## 2. O que já foi escrito no artigo.md

| Seção | Status |
|---|---|
| Cabeçalho (título, autores, afiliações) | ✅ Completo |
| Resumo / Abstract | ❌ Falta |
| **1. Introdução** | ✅ Completo |
| 1.1 Contextualização | ✅ |
| 1.2 Definição do problema | ✅ |
| 1.3 Objetivos | ✅ |
| 1.4 Contribuições | ✅ |
| **2. Referencial Teórico** | ✅ Completo |
| 2.1 IA e Aprendizado de Máquina | ✅ |
| 2.2 NLP | ✅ |
| 2.3 Representação Textual e Modelos | ✅ |
| 2.4 Análise de Sentimentos em Plataformas Digitais | ✅ |
| 2.5 NLP Aplicado à Saúde Mental | ✅ |
| 2.6 LLMs como Anotadores de Dados | ✅ |
| **3. Metodologia** | ⚠️ Parcial |
| 3.1 Visão Geral do Pipeline | ✅ |
| 3.2 Seleção dos Canais | ✅ |
| 3.3 Coleta de Dados (script 00) | ✅ |
| 3.4 Limpeza e Pré-processamento (scripts 01–02) | ✅ |
| 3.5 Rotulação com LLM (script 03) | ✅ |
| 3.6 Amostragem para Revisão Humana (script 04) | ✅ |
| 3.7 Geração de Embeddings BERTimbau (script 05) | ✅ |
| 3.8+ Validação, Treinamento e Análise (scripts 06–09) | ❌ Falta — **sua parte** |
| **4. Resultados e Discussão** | ❌ Falta — **sua parte** |
| **5. Conclusão** | ❌ Falta — **sua parte** |
| **Referências Bibliográficas** | ✅ (geradas automaticamente pelo Pandoc) |

---

## 3. O que falta escrever (sua parte)

### 3.1 Continuação da Metodologia (seções 3.8 em diante)

Você deve descrever as etapas dos **scripts 06 a 09**. Abaixo está o que cada um faz:

#### Script 06 — Validação Cruzada (`06_cv_validation.py`)
- Avalia o pipeline BERTimbau + XGBoost via validação cruzada estratificada **5-fold** sobre os 34.491 comentários não-DESCARTÁVEL
- Objetivo exclusivamente acadêmico: demonstrar capacidade de generalização
- Nenhum modelo é salvo nesta etapa
- Parâmetros XGBoost: `n_estimators=200`, `max_depth=6`, `learning_rate=0.1`, `sample_weight='balanced'`

**Resultados (5-fold CV):**

| Fold | Acurácia | F1-macro | F1-weighted |
|---|---|---|---|
| 1 | 0,8545 | 0,7254 | 0,8522 |
| 2 | 0,8636 | 0,7382 | 0,8609 |
| 3 | 0,8653 | 0,7346 | 0,8623 |
| 4 | 0,8620 | 0,7334 | 0,8598 |
| 5 | 0,8545 | 0,7314 | 0,8530 |
| **Média** | **0,8600 ± 0,005** | **0,7326 ± 0,004** | **0,8576 ± 0,004** |

#### Script 06b — Treinamento e Predição Half-Split (`06b_half_split_evaluation.py`)
- Treina o XGBoost na **1ª metade** (17.772 comentários) e prediz na **2ª metade** (16.719 comentários)
- Avaliação genuína: o modelo nunca viu os dados de teste durante o treinamento
- Gera `confidence` (probabilidade máxima do XGBoost) e `intensidade` (ALTA/MÉDIA/BAIXA)
- Salva o modelo em `XGBoost_half.pkl`

**Divisão dos dados:**

| Conjunto | Total | NEGATIVO | TANGENCIAL | POSITIVO |
|---|---|---|---|---|
| Treino (1ª metade) | 17.772 | 699 (3,9%) | 8.141 (45,8%) | 8.932 (50,3%) |
| Teste (2ª metade) | 16.719 | 1.084 (6,5%) | 6.314 (37,8%) | 9.321 (55,8%) |

**Resultados (half-split — avaliação genuína):**

```
              precision    recall  f1-score   support
    NEGATIVO       0,51      0,24      0,32      1.084
  TANGENCIAL       0,77      0,89      0,82      6.314
    POSITIVO       0,93      0,89      0,91      9.321
    accuracy                           0,85     16.719
   macro avg       0,74      0,67      0,69     16.719
weighted avg       0,84      0,85      0,84     16.719
```

Concordância geral: **14.178 / 16.719 = 84,8%**

**Intensidade emocional (label XGBoost):**

| Intensidade | Critério | Contagem | % |
|---|---|---|---|
| ALTA | confidence > 0,80 | 13.039 | 78,0% |
| MÉDIA | 0,60 ≤ confidence ≤ 0,80 | 2.437 | 14,6% |
| BAIXA | confidence < 0,60 | 1.243 | 7,4% |

**Matriz de confusão** (linhas = LLM, colunas = XGBoost):

| | NEGATIVO | TANGENCIAL | POSITIVO |
|---|---|---|---|
| **NEGATIVO** | 255 | 739 | 90 |
| **TANGENCIAL** | 188 | 5.603 | 523 |
| **POSITIVO** | 59 | 942 | 8.320 |

#### Script 07 — Construção do Dataset Final (`07_build_final_dataset.py`)
- Cruza as predições do XGBoost com os metadados dos comentários
- Calcula coluna `keywords`: termos de credibilidade percebida encontrados no texto
- Gera `final_labeled_dataset.csv` (16.719 linhas)

#### Script 08 — Heurística de Confiança Percebida (`08_credibility_heuristic.py`)
- Classifica cada comentário em ALTA / BAIXA / NEUTRA confiança percebida
- Usa lista de palavras-chave com tratamento de negação (janela de 3 palavras)
- Em caso de conflito (sinais ALTA e BAIXA no mesmo comentário), desambigua via `label_xgb`
- 92% dos comentários resultam em NEUTRA (sem sinal explícito de credibilidade)
- Gera `final_dataset.csv` — dataset principal para análise

#### Script 09 — Análise de Resultados (`09_analyze_results.py`)
- Responde as 6 perguntas de pesquisa usando `final_dataset.csv`
- Testes estatísticos: **qui-quadrado** (P1–P4, P6) e **Cohen's Kappa** (P5)

---

### 3.2 Seção 4 — Resultados e Discussão

Os resultados abaixo são os dados reais do experimento. Use-os diretamente.

**Dataset final:** 16.719 comentários (2ª metade — predições genuínas do XGBoost)

#### P1 — Sentimentos em comentários engajados (likeCount ≥ 10)
n = 323 profissional / 356 amador

| Rótulo | Profissional | Amador |
|---|---|---|
| POSITIVO | 34,67% (112) | 36,52% (130) |
| NEGATIVO | 5,57% (18) | 3,65% (13) |
| TANGENCIAL | 59,75% (193) | 59,83% (213) |

**Chi² = 1,53 — p = 0,4653 — NÃO significativa.**
Interpretação: comentários de alto engajamento apresentam distribuição de sentimentos estatisticamente semelhante entre os dois grupos.

---

#### P2 — Proporção de positivos na base completa

| Rótulo | Profissional | Amador |
|---|---|---|
| POSITIVO | 33,43% (2.296) | 67,37% (6.637) |
| NEGATIVO | 4,86% (334) | 1,71% (168) |
| TANGENCIAL | 61,71% (4.238) | 30,92% (3.046) |

**Chi² = 1.887,33 — p ≈ 0 — Altamente significativa.**
Interpretação: canais amadores concentram muito mais comentários positivos; canais profissionais concentram comentários tangenciais (relatos pessoais e referências ao tema).

---

#### P3 — Intensidade emocional (apenas POSITIVO e NEGATIVO)

**Canais profissionais** (n = 2.630 | pos = 2.296 | neg = 334):

| Intensidade | Total | Positivo | Negativo |
|---|---|---|---|
| ALTA | 73,95% | 80,62% | 28,14% |
| MÉDIA | 16,08% | 12,85% | 38,32% |
| BAIXA | 9,96% | 6,53% | 33,53% |

**Canais amadores** (n = 6.805 | pos = 6.637 | neg = 168):

| Intensidade | Total | Positivo | Negativo |
|---|---|---|---|
| ALTA | 84,10% | 85,61% | 24,40% |
| MÉDIA | 10,29% | 9,45% | 43,45% |
| BAIXA | 5,61% | 4,94% | 32,14% |

**Chi² = 130,13 — p ≈ 0 — Significativa.**
Interpretação: canais amadores geram reações de maior intensidade emocional. Em ambos os grupos, comentários negativos distribuem-se de forma mais equilibrada entre as três faixas, enquanto positivos concentram-se na ALTA intensidade.

---

#### P4 — Confiança percebida (apenas ALTA e BAIXA — exclui NEUTRA)
Total com sinal explícito: 1.311 comentários (8% da base)

| Confiança | Profissional | Amador |
|---|---|---|
| ALTA | 62,24% (455) | 75,69% (439) |
| BAIXA | 37,76% (276) | 24,31% (141) |

**Chi² = 26,34 — p ≈ 0 — Significativa.**
Interpretação: canais amadores apresentam maior proporção de comentários com sinais explícitos de alta confiança percebida. Importante ressaltar que 92% da base é NEUTRA — os percentuais acima referem-se apenas ao subconjunto com sinal explícito de credibilidade.

---

#### P5 — Concordância LLM vs. XGBoost

- Concordância geral: **14.178 / 16.719 = 84,8%**
- **Cohen's Kappa = 0,7163 — Concordância Substancial** (intervalo 0,60–0,80)

Matriz de confusão (linhas = LLM, colunas = XGBoost):

| | POSITIVO | TANGENCIAL | NEGATIVO |
|---|---|---|---|
| **POSITIVO** | 8.320 | 942 | 59 |
| **TANGENCIAL** | 523 | 5.603 | 188 |
| **NEGATIVO** | 90 | 739 | 255 |

Principal padrão de discordância: 739 dos 1.084 comentários que o LLM rotulou como NEGATIVO foram classificados como TANGENCIAL pelo XGBoost (recall de 24% para NEGATIVO).

---

#### P6 — Comentários TANGENCIAIS por grupo
(mesmo dado de P2)

- Profissional: **61,71%** (4.238) | Amador: **30,92%** (3.046)
- **Chi² significativo** — profissionais geram proporcionalmente muito mais relatos pessoais e tangenciais.

---

### 3.3 Seção 5 — Conclusão

Sugestão de estrutura:
- Retomar o objetivo geral e confirmar que foi atingido
- Síntese dos principais achados das 6 perguntas
- Limitações do trabalho (itens abaixo são importantes de mencionar):
  - Heurística de confiança percebida baseada em palavras-chave (insensível a contexto)
  - Classe NEGATIVO com recall baixo (24%) — dificuldade do modelo com classe rara
  - 92% dos comentários resultam em NEUTRA na confiança percebida
  - Corpus limitado ao YouTube brasileiro e aos 12 canais selecionados
- Trabalhos futuros

---

## 4. Regras de escrita a seguir

1. **Use apenas as referências já no `referencias.bib`** — não invente referências novas. As chaves disponíveis são listadas abaixo.
2. **Não invente dados.** Use somente os números desta seção 3 ou os que estiverem nos arquivos CSV em `data/processed/results/`.
3. **Tempo verbal:** metodologia no passado ("foram coletados", "utilizou-se"); resultados no passado ou presente ("os resultados indicam", "observou-se que"); conclusão no presente.
4. **Estilo:** seguir o tom do que já foi escrito — formal, acadêmico, sem bullet points no corpo do texto.
5. O TODO no final da seção 3 deve ser **removido** quando você completar as seções 3.8+.

---

## 5. Referências disponíveis no referencias.bib

| Chave | Referência |
|---|---|
| `[@afonso2019]` | Afonso & Duque (2019) — análise de sentimentos em YouTube, SVM + TF-IDF |
| `[@bojanowski2016]` | Bojanowski et al. (2016) — FastText, word vectors com subword |
| `[@devlin2019]` | Devlin et al. (2019) — BERT |
| `[@goodfellow2016]` | Goodfellow, Bengio & Courville (2016) — Deep Learning (livro) |
| `[@howard2018]` | Howard & Ruder (2018) — ULMFiT, transfer learning em NLP |
| `[@jurafsky2023]` | Jurafsky & Martin (2023) — Speech and Language Processing (livro) |
| `[@li2021]` | Li et al. (2021) — survey: text classification, shallow to deep learning |
| `[@mikolov2013]` | Mikolov et al. (2013) — Word2Vec |
| `[@pennington2014]` | Pennington, Socher & Manning (2014) — GloVe |
| `[@russell2021]` | Russell & Norvig (2021) — Artificial Intelligence: A Modern Approach (livro) |
| `[@souza2020]` | Souza (2020) — BERTimbau |
| `[@souza2021]` | Souza & Filho (2021) — análise de sentimentos em português brasileiro |
| `[@wang2023]` | Wang et al. (2023) — survey: sentiment analysis |
| `[@ding2023]` | Ding et al. (2023) — Is GPT-3 a Good Data Annotator? (ACL 2023) |
| `[@tan2024]` | Tan et al. (2024) — LLMs for Data Annotation: A Survey (EMNLP 2024) |
| `[@anthropic2024]` | Anthropic (2024) — The Claude 3 Model Family (model card) |

---

## 6. Sugestão de prompt inicial para o Claude

Ao abrir uma nova conversa com o Claude, cole algo como:

> "Estou escrevendo o TCC Parte 2 no formato artigo Mackenzie. O arquivo principal é `artigo.md` e as referências estão em `referencias.bib`. Leia o arquivo `HANDOFF_COAUTORA.md` para entender o que já foi feito e o que falta. Quero sua ajuda para escrever [seção X]."

Em seguida, compartilhe o conteúdo deste arquivo e do `artigo.md` atual para que o Claude tenha contexto completo.
