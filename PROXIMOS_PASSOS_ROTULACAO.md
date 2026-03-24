# Próximos Passos: Da Limpeza à Rotulação Completa — TCC

> **Onde estamos agora:** Dados limpos e pré-processados em `data/processed/`.
> Este documento descreve tudo que precisa acontecer até termos um dataset
> totalmente rotulado, pronto para treinar o BERTimbau.

---

## Sumário

1. [O que é rotulação e por que precisamos dela](#1-o-que-é-rotulação-e-por-que-precisamos-dela)
2. [Decisão crítica: binário ou ternário?](#2-decisão-crítica-binário-ou-ternário)
3. [O problema de rotular 14.760 comentários](#3-o-problema-de-rotular-14760-comentários)
4. [Estratégia recomendada: rotulação híbrida](#4-estratégia-recomendada-rotulação-híbrida)
5. [Fase 1 — Guia de anotação](#5-fase-1--guia-de-anotação)
6. [Fase 2 — Rotulação automática como ponto de partida](#6-fase-2--rotulação-automática-como-ponto-de-partida)
7. [Fase 3 — Rotulação via LLM (recomendada)](#7-fase-3--rotulação-via-llm-recomendada)
8. [Fase 4 — Validação humana e concordância](#8-fase-4--validação-humana-e-concordância)
9. [Fase 5 — Dataset final rotulado](#9-fase-5--dataset-final-rotulado)
10. [Diagrama do fluxo completo](#10-diagrama-do-fluxo-completo)
11. [Resumo de tecnologias e dependências](#11-resumo-de-tecnologias-e-dependências)

---

## 1. O que é rotulação e por que precisamos dela

O BERTimbau é um modelo de **aprendizado supervisionado**. Isso significa que,
para ensiná-lo a classificar sentimentos, precisamos de exemplos onde **já
sabemos a resposta certa**: "este comentário é positivo", "aquele é negativo".

Esses exemplos com respostas conhecidas se chamam **dados rotulados** (ou
*labeled data*). Sem eles, o modelo não tem como aprender.

No nosso caso:
- Temos **14.760 comentários** sem rótulo.
- Precisamos atribuir a cada um a etiqueta `positivo` ou `negativo`.
- Depois, dividimos em treino/teste e treinamos o BERTimbau.

**A qualidade da rotulação determina diretamente a qualidade do modelo.**
Um dataset mal rotulado produz um modelo ruim mesmo com a melhor arquitetura.

---

## 2. Decisão crítica: binário ou ternário?

O relatório do TCC define classificação **binária** (positivo/negativo).
Antes de começar a rotular, vale avaliar se isso é suficiente.

### Opção A — Binário: `positivo` / `negativo`

| Vantagens | Desvantagens |
|-----------|--------------|
| Mais simples de rotutar | Comentários neutros precisam ser forçados em um dos lados |
| Modelo mais fácil de treinar | Pode criar ruído no dataset |
| Alinhado com o que o relatório descreve | Pergunta de pesquisa pode ser menos rica |

**Exemplo problemático:** `"Interessante, não sabia disso."` — neutro de sentimento,
mas você precisaria escolher positivo ou negativo.

### Opção B — Ternário: `positivo` / `negativo` / `neutro`

| Vantagens | Desvantagens |
|-----------|--------------|
| Mais realista para comentários do YouTube | Mais difícil de rotutar (onde está a fronteira?) |
| Não força rótulo em comentários informativos | Modelo mais complexo |
| Permite análise mais nuançada | Requer mais exemplos de treinamento |

### Recomendação

> **Use binário**, conforme o relatório, mas **crie uma terceira categoria
> temporária `"descartável"`** para comentários que não expressam sentimento
> algum (ex: perguntas factuais, spam, timestamps como `"0:45"`). Esses são
> excluídos do treino. Isso é diferente de "neutro" — é simplesmente texto sem
> opinião emocional sobre o conteúdo.

---

## 3. O problema de rotular 14.760 comentários

Rotular tudo à mão é inviável para duas pessoas:

| Cenário | Tempo estimado |
|---------|---------------|
| 2 min por comentário, 2 anotadoras | ~246 horas cada (~30 semanas em tempo parcial) |
| 30 seg por comentário, 2 anotadoras | ~61 horas cada (~7 semanas em tempo parcial) |

Além do tempo, rotular tudo manualmente seria **cansativo** e levaria a erros e
inconsistências no final. A solução é combinar automação com revisão humana
focada.

A boa notícia: **não precisamos rotular tudo manualmente**. Precisamos de um
subconjunto de alta qualidade para treinar e avaliar o modelo, e podemos usar
ferramentas automáticas para o resto.

---

## 4. Estratégia recomendada: rotulação híbrida

```
Dataset bruto (14.760 comentários)
         │
         ▼
  [Rotulação automática]   ←── LeIA + pysentimiento como baseline
         │
         ▼
  [Rotulação via LLM]      ←── Claude Haiku via API (~R$ 3-10 total)
         │                      Alta precisão, muito rápido
         ▼
  [Validação humana]       ←── ~500 comentários revisados manualmente
         │                      (foco nos casos difíceis e incertos)
         ▼
  [Cálculo de concordância] ←── Krippendorff's Alpha ≥ 0,70
         │
         ▼
  Dataset rotulado final
  (~14.000-14.500 comentários com rótulo confiável)
```

Esta abordagem é usada em pesquisa acadêmica recente (2024-2025) e oferece o
melhor equilíbrio entre custo, velocidade e qualidade.

---

## 5. Fase 1 — Guia de anotação

Antes de rotular qualquer coisa, precisamos escrever **regras claras** para
decidir o que é positivo ou negativo. Sem isso, duas pessoas rotulam o mesmo
comentário de formas diferentes.

### O que significa "positivo" neste contexto?

O sentimento não é "o comentário é agradável de ler" — é **a reação emocional
do espectador ao conteúdo do vídeo**.

| Tipo de reação | Rótulo | Exemplo |
|----------------|--------|---------|
| Gratidão, alívio, apoio | `positivo` | `"Esse vídeo me ajudou tanto, obrigada 💛"` |
| Identificação positiva | `positivo` | `"Exatamente o que eu precisava ouvir"` |
| Esperança, motivação | `positivo` | `"Depois de assistir isso fui buscar ajuda"` |
| Elogio ao criador | `positivo` | `"Você explica melhor que minha terapeuta"` |
| Tristeza, angústia expressada | `negativo` | `"Fico mal toda vez que vejo conteúdo assim"` |
| Crítica ao conteúdo | `negativo` | `"Isso é simplificação demais, perigoso"` |
| Desconforto, trauma ativado | `negativo` | `"Não consigo nem terminar de assistir"` |
| Discordância forte | `negativo` | `"Isso não é assim, é desinformação"` |
| Pergunta factual sem emoção | `descartável` | `"Qual o nome do livro que você citou?"` |
| Spam, promoção | `descartável` | `"Acesse meu canal também!"` |
| Timestamp | `descartável` | `"2:34 melhor parte"` |
| Só emojis ambíguos | `descartável` | `"👀"` |

### Casos difíceis — como resolver

**Misto (positivo + negativo no mesmo comentário):**
> `"Conteúdo incrível mas fico triste sempre que lembro de tudo isso"`

→ Use o sentimento **dominante** ou o da **última oração** (mais recente
emocionalmente). Se não der para decidir, marque como `descartável`.

**Ironia/sarcasmo:**
> `"Sim, claro, 'só pense positivo' resolve depressão"`

→ O sentimento superficial é positivo, mas o real é negativo (crítica). Marque
como `negativo`. Esses casos são raros mas importantes de documentar no guia.

**Emoção sobre a própria situação vs. sobre o vídeo:**
> `"Estou passando por um momento muito difícil"`

→ O espectador expressa sofrimento, mas não está reagindo especificamente ao
vídeo. Marque como `negativo` (o conteúdo ativou/conectou a essa emoção).

> **Dica:** Salve os exemplos difíceis que surgirem durante a anotação em um
> arquivo `guia_anotacao.md`. Isso vai crescer e ajudar a manter consistência.

---

## 6. Fase 2 — Rotulação automática como ponto de partida

Antes de gastar dinheiro com LLM, use ferramentas gratuitas para ter uma
**linha de base** e entender como o dataset se distribui.

### Ferramenta 1: LeIA (Léxico para Inferência Adaptada)

LeIA é uma adaptação do VADER para português brasileiro. Funciona bem com
texto informal e emojis — ideal para comentários do YouTube.

```bash
pip install leia-br
```

```python
from leiapt import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()
scores = sia.polarity_scores("Esse vídeo me ajudou muito, obrigada! 💛")
# {'neg': 0.0, 'neu': 0.421, 'pos': 0.579, 'compound': 0.7003}
# compound > 0.05 → positivo
# compound < -0.05 → negativo
# entre -0.05 e 0.05 → neutro/descartável
```

**Limitação:** Baseada em dicionário. Não entende contexto. `"Não estou bem"` pode
ser classificado errado porque `"bem"` tem score positivo.

### Ferramenta 2: pysentimiento

Biblioteca que usa modelos Transformer para análise de sentimento em redes
sociais, com suporte a português.

```bash
pip install pysentimiento
```

```python
from pysentimiento import create_analyzer

analyzer = create_analyzer(task="sentiment", lang="pt")
result = analyzer.predict("Esse vídeo me ajudou muito!")
# SentimentOutput(output='POS', probas={'POS': 0.92, 'NEG': 0.04, 'NEU': 0.04})
```

**Vantagem sobre LeIA:** Entende contexto linguístico. **Desvantagem:** Mais lento
e pode não ser especializado em saúde mental.

### O que fazer com os resultados automáticos

Use os dois ferramentas e compare:

```python
# Caso os dois concordam → alta confiança, pode usar como rótulo provisional
# Caso discordam → baixa confiança, priorize para revisão humana
```

**Script a criar:** `scripts/03_auto_label.py` — que aplica LeIA e pysentimiento
a `preprocessed_comments.csv` e salva colunas `label_leia` e `label_pysent`,
além de um score de concordância.

---

## 7. Fase 3 — Rotulação via LLM (recomendada)

Esta é a abordagem com **melhor custo-benefício** para o tamanho do dataset.
Pesquisas de 2024-2025 mostram que LLMs superam anotadores humanos do Amazon
Mechanical Turk em tarefas de classificação de sentimento.

### Por que usar Claude Haiku (API)?

| Modelo | Precisão estimada | Custo por 15K comentários |
|--------|-------------------|---------------------------|
| Claude Haiku | Alta (~85-90%) | ~R$ 3-8 |
| GPT-4o-mini | Alta (~82-88%) | ~R$ 5-12 |
| LeIA (grátis) | Média (~65-75%) | Grátis |
| pysentimiento | Média-Alta (~75-82%) | Grátis |

Claude Haiku custa US$ 0,25 por milhão de tokens de entrada. Nosso dataset tem
~14.760 comentários com média de ~100 palavras = ~1,5 milhão de tokens.
**Custo total: ~US$ 0,50-1,50 (cerca de R$ 3-9).**

### Estratégia de prompt

A qualidade do rótulo depende muito do prompt. Use **few-shot prompting** (dê
exemplos ao modelo) e seja específico sobre o contexto:

```python
SYSTEM_PROMPT = """Você é um especialista em análise de sentimentos aplicada a
saúde mental. Sua tarefa é classificar comentários de vídeos do YouTube sobre
saúde mental e bem-estar no Brasil.

Classifique cada comentário em UMA das três categorias:
- POSITIVO: o comentário expressa emoção positiva em relação ao conteúdo
  (gratidão, alívio, identificação, esperança, apoio)
- NEGATIVO: o comentário expressa emoção negativa em relação ao conteúdo
  (tristeza, angústia, crítica, discordância, desconforto)
- DESCARTAVEL: o comentário não expressa sentimento emocional claro
  (perguntas factuais, spam, timestamps, emojis ambíguos)

Responda APENAS com uma dessas palavras: POSITIVO, NEGATIVO ou DESCARTAVEL.
Não adicione explicações."""

FEW_SHOT_EXAMPLES = [
    ("Esse vídeo me fez chorar de alívio. Obrigada por falar sobre isso 💛", "POSITIVO"),
    ("Discordo completamente. Isso é uma simplificação perigosa da depressão.", "NEGATIVO"),
    ("Qual o nome do livro que você mencionou aos 3:45?", "DESCARTAVEL"),
    ("Não consigo nem terminar de assistir porque me traz muitas memórias ruins", "NEGATIVO"),
    ("Finalmente alguém explicando isso de forma acessível!", "POSITIVO"),
    ("👀", "DESCARTAVEL"),
]
```

**Script a criar:** `scripts/04_llm_label.py` — que usa a API do Claude para
rotular em lotes (*batch processing*) e salva `label_llm` com a confiança.

> **Observação:** Para usar a API do Claude, você precisará de uma conta
> Anthropic e uma chave de API (ANTHROPIC_API_KEY). O custo total é menor que
> uma pizza.

### Processamento em lote (para economizar)

Em vez de chamar a API para cada comentário individualmente (14.760 chamadas),
agrupe os comentários em lotes de 20-50 por chamada:

```python
# Em vez de:
# "Classifique este comentário: {comentario}" × 14.760 vezes

# Faça:
# "Classifique os seguintes 30 comentários, um por linha..." × ~490 chamadas
```

Isso reduz custos em ~60% e é mais rápido.

---

## 8. Fase 4 — Validação humana e concordância

### Quantos comentários revisar manualmente?

**Recomendação:** 500-800 comentários, selecionados estrategicamente:

| Critério de seleção | Quantidade | Por quê |
|--------------------|------------|---------|
| Discordância LeIA vs. LLM | ~200 | São os casos mais incertos |
| Amostra aleatória geral | ~150 | Para estimar acurácia geral |
| Comentários só-emoji | ~50 | Caso especial frequente |
| Comentários muito curtos (< 10 chars) | ~50 | Difíceis para modelos |
| Comentários muito longos (> 500 chars) | ~50 | Risco de sentimento misto |
| Alta relevância (likeCount > 100) | ~100 | Alta influência no dataset |
| **Total** | **~600** | ~4% do dataset |

### Ferramenta recomendada: Label Studio (gratuito)

Label Studio é uma ferramenta open-source para anotação de dados com interface
visual. Ideal para duas anotadoras trabalhando em paralelo.

**Instalação e uso:**

```bash
pip install label-studio
label-studio start
# Acessa via navegador em http://localhost:8080
```

**Configuração para o projeto:**
1. Crie um projeto "TCC — Sentimentos Saúde Mental"
2. Configure interface de classificação (3 botões: Positivo / Negativo / Descartável)
3. Importe os ~600 comentários selecionados como CSV
4. Cada anotadora rotula o mesmo conjunto (para calcular concordância)
5. Exporte os rótulos em JSON/CSV

**Alternativa mais simples:** Uma planilha Google Sheets compartilhada com as
colunas `texto`, `rótulo_ana`, `rótulo_outra_pessoa`. Funciona bem para 600
comentários.

### Calculando concordância entre anotadoras

Depois que ambas terminarem, calcule o **Krippendorff's Alpha** para medir
o quanto as anotações concordam:

```bash
pip install krippendorff
```

```python
import krippendorff
import numpy as np

# 0 = DESCARTAVEL, 1 = NEGATIVO, 2 = POSITIVO
rotulos_ana    = [2, 1, 2, 0, 1, ...]  # rótulos da Ana
rotulos_colega = [2, 1, 0, 0, 1, ...]  # rótulos da colega

alpha = krippendorff.alpha(
    reliability_data=[rotulos_ana, rotulos_colega],
    level_of_measurement="nominal"
)
print(f"Krippendorff's Alpha: {alpha:.3f}")
```

**Interpretação:**

| Alpha      | Interpretação | O que fazer |
|------------|---------------|-------------|
| > 0,80     | Excelente     | Pode usar os rótulos com confiança |
| 0,67 – 0,80| Bom           | Aceitável para publicação acadêmica |
| 0,50 – 0,67| Moderado      | Resolver discordâncias antes de usar |
| < 0,50     | Fraco         | Revisar guia de anotação e refazer |

**Meta para este TCC:** Alpha ≥ 0,70.

Quando houver discordância, as duas anotadoras discutem o caso e chegam a um
consenso. O consenso vira o rótulo final.

---

## 9. Fase 5 — Dataset final rotulado

### Consolidação dos rótulos

A lógica de decisão final:

```
Para cada comentário:
  Se foi revisado manualmente → usa o rótulo humano (consenso)
  Senão, se LLM rotulou → usa o rótulo do LLM
  Senão, se LeIA e pysentimiento concordam → usa o rótulo deles
  Senão → marca como "incerto" e considera descartar do treino
```

### Estrutura do dataset final

O arquivo `data/processed/labeled_comments.csv` terá as colunas:

| Coluna | Descrição |
|--------|-----------|
| `commentId` | ID único |
| `text` | Texto original |
| `text_clean` | Texto limpo (para BERTimbau) |
| `text_preprocessed` | Texto pré-processado (para TF-IDF) |
| `channel_type` | `profissional` ou `amador` |
| `likeCount` | Curtidas |
| `label` | Rótulo final: `positivo`, `negativo` ou `descartavel` |
| `label_source` | De onde veio: `manual`, `llm`, `auto` |
| `label_confidence` | Confiança (0.0–1.0, NaN se manual) |

### Distribuição esperada

Com base em análise de comentários de saúde mental no YouTube (literatura):
- ~60-70% positivos (YouTube tem viés positivo: comentários negativos são
  frequentemente removidos ou menos frequentes em canais de bem-estar)
- ~20-30% negativos
- ~10-15% descartáveis

> **Atenção ao desbalanceamento:** Se a proporção positivo/negativo for muito
> diferente (ex: 80%/20%), as métricas de acurácia vão enganar. Use F1-score
> como métrica principal e considere técnicas como oversampling (SMOTE) ou
> pesos de classe no treinamento.

---

## 10. Diagrama do fluxo completo

```
[data/processed/preprocessed_comments.csv]  ← já temos isso
              │
              ▼
    ┌─────────────────────┐
    │  03_auto_label.py   │  LeIA + pysentimiento
    │  (grátis, rápido)   │  → label_leia, label_pysent
    └────────┬────────────┘
             │
             ▼
    ┌─────────────────────┐
    │  04_llm_label.py    │  Claude Haiku API (~R$5)
    │  (alta qualidade)   │  → label_llm, llm_confidence
    └────────┬────────────┘
             │
             ▼
    ┌─────────────────────┐
    │  Seleção inteligente│  ~600 comentários incertos/importantes
    │  para revisão humana│  para revisão manual
    └────────┬────────────┘
             │
             ▼
    ┌─────────────────────┐
    │  Label Studio /     │  2 anotadoras rotulam os ~600
    │  Google Sheets      │  → concordância Krippendorff ≥ 0,70
    └────────┬────────────┘
             │
             ▼
    ┌─────────────────────┐
    │  05_build_dataset.py│  Consolida todos os rótulos
    │  (consolidação)     │  → labeled_comments.csv
    └────────┬────────────┘
             │
             ▼
    [data/processed/labeled_comments.csv]
    ~14.000 comentários rotulados
    Prontos para treinar o BERTimbau
```

---

## 11. Resumo de tecnologias e dependências

### Instalar tudo de uma vez

```bash
pip install leia-br pysentimiento krippendorff label-studio anthropic
```

### Tabela de ferramentas

| Ferramenta | Uso | Custo | Instalar |
|------------|-----|-------|---------|
| **LeIA** | Rotulação automática baseline | Gratuito | `pip install leia-br` |
| **pysentimiento** | Rotulação automática com Transformer | Gratuito | `pip install pysentimiento` |
| **Claude Haiku API** | Rotulação de alta qualidade | ~R$ 3-10 | `pip install anthropic` |
| **Label Studio** | Interface visual de anotação | Gratuito | `pip install label-studio` |
| **krippendorff** | Cálculo de concordância | Gratuito | `pip install krippendorff` |

### Próximos scripts a criar

| Script | O que faz |
|--------|-----------|
| `scripts/03_auto_label.py` | Aplica LeIA e pysentimiento ao dataset |
| `scripts/04_llm_label.py` | Usa Claude API para rotular em lote |
| `scripts/05_build_dataset.py` | Consolida rótulos e gera dataset final |

---

## Checklist de execução

```
[ ] Fase 1: Escrever guia_anotacao.md com exemplos de cada categoria
[ ] Fase 2: Criar e rodar scripts/03_auto_label.py (LeIA + pysentimiento)
[ ] Fase 2: Analisar distribuição dos rótulos automáticos
[ ] Fase 3: Criar conta Anthropic e obter ANTHROPIC_API_KEY
[ ] Fase 3: Criar e rodar scripts/04_llm_label.py com Claude Haiku
[ ] Fase 4: Selecionar ~600 comentários para revisão humana
[ ] Fase 4: Configurar Label Studio ou planilha compartilhada
[ ] Fase 4: Ambas as anotadoras rotulam os ~600 comentários
[ ] Fase 4: Calcular Krippendorff's Alpha (meta: ≥ 0,70)
[ ] Fase 4: Resolver discordâncias por consenso
[ ] Fase 5: Criar e rodar scripts/05_build_dataset.py
[ ] Fase 5: Verificar distribuição final positivo/negativo
[ ] Fase 5: Dataset rotulado pronto → começar embeddings com BERTimbau
```

---

*Documento escrito em 24/03/2026. Baseado na proposta do TCC e em literatura
recente (2024-2025) sobre rotulação de dados NLP e análise de sentimentos
em português.*
