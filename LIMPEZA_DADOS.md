# Limpeza e Pré-processamento de Dados — TCC

> **Contexto:** Este documento descreve as decisões e etapas de limpeza aplicadas
> aos dados coletados via YouTube Data API para o TCC _"Análise de impacto dos
> vídeos de saúde e bem-estar"_ (Mackenzie, 2025).

---

## Sumário

1. [Visão geral dos dados brutos](#1-visão-geral-dos-dados-brutos)
2. [Scripts criados](#2-scripts-criados)
3. [Script 1: Limpeza estrutural](#3-script-1-limpeza-estrutural-01_clean_datapy)
4. [Script 2: Pré-processamento de texto](#4-script-2-pré-processamento-de-texto-02_preprocess_textpy)
5. [Por que duas versões de texto?](#5-por-que-duas-versões-de-texto)
6. [Resultados: antes e depois](#6-resultados-antes-e-depois)
7. [Exemplos de transformação](#7-exemplos-de-transformação)
8. [Arquivos gerados](#8-arquivos-gerados)
9. [Como re-executar](#9-como-re-executar)

---

## 1. Visão geral dos dados brutos

Antes de qualquer limpeza, os dois CSVs apresentavam a seguinte estrutura:

### `raw_comments.csv` — 14.762 linhas, 7 colunas

| Coluna        | Tipo    | Descrição                                                    |
|---------------|---------|--------------------------------------------------------------|
| `videoId`     | string  | ID único do vídeo no YouTube                                 |
| `video_title` | string  | Título do vídeo                                              |
| `commentId`   | string  | ID único do comentário                                       |
| `publishedAt` | string  | Data/hora de publicação (formato ISO 8601)                   |
| `text`        | string  | Texto do comentário                                          |
| `likeCount`   | inteiro | Número de curtidas no comentário                             |
| `channel_type`| string  | Tipo do canal: `"profissional"` ou `"amador"`               |

**Problemas identificados antes da limpeza:**

| Problema                                | Quantidade |
|-----------------------------------------|------------|
| Textos nulos (`NaN`)                    | 2          |
| Comentários duplicados (`commentId`)    | 0          |
| Datas como string (não como datetime)   | 14.762     |
| Comentários só com emojis (sem palavras)| ~1.479     |
| Comentários com URLs                    | 36         |
| Comentários com menções (`@usuario`)    | 35         |
| Comentários com hashtags                | 27         |

### `video_metadata.csv` — 40 linhas, 9 colunas

| Coluna         | Tipo    | Descrição                                        |
|----------------|---------|--------------------------------------------------|
| `videoId`      | string  | ID único do vídeo                                |
| `title`        | string  | Título do vídeo                                  |
| `description`  | string  | Descrição do vídeo                               |
| `channelId`    | string  | ID do canal                                      |
| `channelTitle` | string  | Nome do canal                                    |
| `publishedAt`  | string  | Data de publicação                               |
| `viewCount`    | inteiro | Visualizações                                    |
| `likeCount`    | inteiro | Curtidas                                         |
| `commentCount` | inteiro | Número de comentários                            |

**Problemas identificados:** 2 descrições nulas, sem coluna `channel_type`.

Os 40 vídeos estão distribuídos em 10 canais (4 vídeos por canal), sendo 5 canais
profissionais e 5 amadores, como definido na proposta do TCC.

---

## 2. Scripts criados

Foram criados dois scripts em `scripts/`, projetados para serem **re-executáveis**:
basta rodar de novo após recoletar dados e o resultado será sempre consistente.

```
scripts/
├── 01_clean_data.py        # Limpeza estrutural (roda em ~2s)
└── 02_preprocess_text.py   # Pré-processamento NLP (roda em ~60s)

data/
└── processed/
    ├── cleaned_comments.csv      # Saída do script 1
    ├── cleaned_metadata.csv      # Saída do script 1
    └── preprocessed_comments.csv # Saída do script 2
```

---

## 3. Script 1: Limpeza estrutural (`01_clean_data.py`)

Este script trata **problemas de qualidade independentes de NLP** — coisas que
precisariam ser corrigidas mesmo que o projeto não fosse de análise de sentimentos.

### Para `raw_comments.csv`

#### Etapa 1 — Remoção de textos nulos
Dois comentários não tinham texto (valor `NaN`). Eles foram removidos porque um
comentário sem texto não pode ser analisado por nenhum modelo.

> **Decisão:** remover em vez de preencher. Não há como inferir o texto original;
> qualquer preenchimento artificial criaria ruído.

#### Etapa 2 — Remoção de textos vazios após strip
Comentários que eram apenas espaços ou quebras de linha foram descartados pelo
mesmo motivo acima.

#### Etapa 3 — Remoção de `commentId` duplicados
Verificação preventiva. No dataset atual não havia duplicatas, mas a checagem
existe para quando dados novos forem coletados.

#### Etapa 4 — Conversão de `publishedAt` para datetime
A data estava armazenada como texto (`"2026-03-14T15:35:12Z"`). Converter para
o tipo `datetime` com fuso UTC permite ordenar cronologicamente, calcular
intervalos (ex: engajamento ao longo do tempo) e facilita joins com o dataset
de metadados.

#### Etapa 5 — Garantia de `likeCount` não-negativo
Verificação de integridade: curtidas não podem ser negativas. Nenhum caso foi
encontrado, mas a regra protege contra erros futuros da API.

#### Etapa 6 — Coluna `text_length`
Número de caracteres do texto original. Útil para análises exploratórias e para
detectar comentários anormalmente longos (possível spam ou copiar-colar).

#### Etapa 7 — Coluna `is_substantive`
`True` quando o comentário contém ao menos uma letra (a-z, A-Z, incluindo
acentuação portuguesa). `False` para comentários que são apenas emojis.

> **Por que não descartar os emojis?** Comentários como `"💛💛💛"` são muito
> frequentes (329 ocorrências de `💛` isolado) e **carregam sentimento claro**
> (positivo). Descartá-los agora significaria perder dados de sentimento válidos.
> A flag `is_substantive` permite que cada etapa posterior decida como tratá-los:
> o BERTimbau consegue processar emojis convertidos em texto, mas o modelo
> TF-IDF convencional não se beneficia deles.

### Para `video_metadata.csv`

- **Descrições nulas:** preenchidas com string vazia (`""`). Ao contrário de
  comentários, vídeos podem existir sem descrição — é um campo opcional.
- **Conversão de datas:** mesmo critério aplicado aos comentários.
- **Coluna `channel_type`:** adicionada com base no mapeamento explícito da
  proposta do TCC (os 5 canais profissionais vs. os 5 amadores). Isso permite
  filtrar e comparar os metadados por tipo de produtor.

---

## 4. Script 2: Pré-processamento de texto (`02_preprocess_text.py`)

Lê `cleaned_comments.csv` e adiciona **duas novas colunas de texto tratado**,
uma para cada abordagem de modelagem prevista no TCC.

### Coluna `text_clean` — para BERTimbau

Aplica apenas limpeza de **ruídos técnicos**, sem alterar a estrutura linguística:

| Transformação                    | Exemplo antes           | Exemplo depois                      |
|----------------------------------|-------------------------|-------------------------------------|
| Decodifica entidades HTML        | `&amp;`                 | `&`                                 |
| Remove URLs                      | `https://bit.ly/xyz`    | _(removido)_                        |
| Remove menções                   | `@fulano obrigado`      | `obrigado`                          |
| Remove hashtags                  | `#saudemental`          | _(removido)_                        |
| Converte emojis em descrição     | `💛`                    | `:coração_amarelo:`                 |
| Normaliza espaços extras         | `"  oi  \n  mundo  "`  | `"oi mundo"`                        |

> **Por que converter emojis em texto em vez de removê-los?**
> O BERTimbau foi treinado com texto da web real, e as descrições textuais de
> emojis (`":coração_amarelo:"`, `":coração_partido:"`) carregam carga semântica
> que o modelo consegue interpretar no contexto da frase. Remover totalmente um
> emoji como `💔` descartaria um indicador de sentimento negativo forte.

### Coluna `text_preprocessed` — para TF-IDF e modelos tradicionais

Aplica normalização linguística completa, conforme descrito na seção 3.2 do
relatório do TCC:

| Etapa                   | Descrição                                                      |
|-------------------------|----------------------------------------------------------------|
| 1. Limpeza técnica      | Mesma de `text_clean`, mas remove emojis completamente         |
| 2. Minúsculas           | `"Ansiedade"` → `"ansiedade"`                                  |
| 3. Remoção de acentos   | `"coração"` → `"coracao"` (normalização NFKD)                 |
| 4. Remoção de não-letras| Pontuação e números eliminados                                 |
| 5. Tokenização          | Divide em palavras usando o tokenizador NLTK para português    |
| 6. Remoção de stopwords | Remove palavras sem carga semântica ("de", "para", "que"...)   |
| 7. Stemming (RSLP)      | Reduz palavras à raiz: `"correndo"` → `"corr"`                 |

**Stopwords:** lista do NLTK para português (`nltk.corpus.stopwords`), que inclui
artigos, preposições, conjunções e pronomes comuns.

**Stemmer RSLP** (Removedor de Sufixos da Língua Portuguesa): algoritmo
desenvolvido especificamente para o português. É mais agressivo que a
lematização — retorna raízes morfológicas (ex: `"ansiedade"` → `"ansiedad"`),
mas é rápido e não exige modelos pesados.

> **Nota sobre lematização real:** O relatório do TCC menciona lematização.
> O RSLP faz *stemming*, que é uma aproximação mais simples. Para lematização
> real (que retorna a forma de dicionário, ex: `"correndo"` → `"correr"`),
> seria necessário instalar o modelo spaCy para português (`pt_core_news_sm`,
> ~500 MB). Para os fins deste TCC — especialmente porque o BERTimbau será o
> modelo principal — o RSLP é suficiente como baseline para os experimentos com
> TF-IDF.

---

## 5. Por que duas versões de texto?

Esta é a decisão mais importante desta etapa, e merece explicação detalhada.

O BERTimbau é um modelo do tipo **Transformer**, treinado em bilhões de palavras
de texto real em português. Ele aprendeu que `"Estou com saudades 💔"` transmite
tristeza *exatamente por causa da pontuação, das maiúsculas e do emoji juntos*.

Se aplicarmos o pré-processamento clássico (minúsculas + sem pontuação + sem
stopwords + RSLP) antes de passar ao BERTimbau, o texto vira `"saudad"` — uma
raiz truncada que o modelo nunca viu em seu treinamento. O resultado é pior.

Já modelos como TF-IDF + Regressão Logística trabalham com **frequência de
palavras**. Para eles, `"correr"`, `"correndo"`, `"corro"` são três tokens
completamente diferentes, o que aumenta a esparsidade (o vocabulário fica grande
demais) sem ganho semântico. O stemming reduz tudo a `"corr"` e o vocabulário
fica compacto e eficiente.

| Abordagem       | Use `text_clean` | Use `text_preprocessed` |
|-----------------|------------------|--------------------------|
| BERTimbau       | ✅                | ❌ (piora o modelo)       |
| TF-IDF + LogReg | ❌                | ✅                        |
| TF-IDF + XGBoost| ❌                | ✅                        |
| Random Forest   | ❌                | ✅                        |

---

## 6. Resultados: antes e depois

### Comentários

| Métrica                               | Valor     |
|---------------------------------------|-----------|
| Linhas no CSV bruto                   | 14.762    |
| Removidos (texto nulo)                | 2         |
| **Linhas em `cleaned_comments.csv`**  | **14.760**|
| Comentários com palavra (is_substantive = True) | 13.281 (89,9%) |
| Comentários só-emoji (is_substantive = False)   | 1.479 (10,0%)  |
| Canais amadores                       | 7.805 comentários |
| Canais profissionais                  | 6.955 comentários |

### Comprimento médio dos comentários por tipo de canal

| Canal         | Média de caracteres | Mediana | Mínimo | Máximo |
|---------------|--------------------:|--------:|-------:|-------:|
| Amador        | 120                 | 48      | 1      | 6.214  |
| Profissional  | 156                 | 74      | 1      | 6.580  |

> Comentários em canais profissionais são, em média, 30% mais longos que os de
> canais amadores. Isso pode indicar engajamento mais reflexivo ou discussões
> mais elaboradas — hipótese interessante para a análise qualitativa.

### Impacto do pré-processamento NLP

| Versão              | Comentários com texto vazio após processamento |
|---------------------|------------------------------------------------|
| `text_clean`        | 16 (comentários que eram só URLs/menções)      |
| `text_preprocessed` | 1.535 (comentários emoji-only + muito curtos) |

Os 1.535 comentários "vazios" em `text_preprocessed` **ainda existem no dataset**
com a coluna `text_clean` preenchida. A decisão de incluí-los ou não no
treinamento dos modelos tradicionais fica para a etapa de modelagem.

### Metadados

| Métrica                   | Valor |
|---------------------------|-------|
| Vídeos no CSV bruto       | 40    |
| Canais analisados         | 10    |
| Vídeos por canal          | 4     |
| Descrições nulas          | 2 (preenchidas com `""`) |
| **Linhas em `cleaned_metadata.csv`** | **40** |

---

## 7. Exemplos de transformação

### Texto com emoji emocional

| Versão             | Texto |
|--------------------|-------|
| Original           | `Estou com saudades 💔` |
| `text_clean`       | `Estou com saudades :coração_partido:` |
| `text_preprocessed`| `saudad` |

### Texto com URL e @menção

| Versão             | Texto |
|--------------------|-------|
| Original           | `@fulano veja aqui: https://t.co/xyz #saúdemental` |
| `text_clean`       | `veja aqui:` |
| `text_preprocessed`| `vej` |

### Texto substantivo longo

| Versão             | Texto (truncado) |
|--------------------|-------|
| Original           | `Não é o poder econômico. É o poder de fato. É o poder da reputação...` |
| `text_clean`       | `Não é o poder econômico. É o poder de fato. É o poder da reputação...` |
| `text_preprocessed`| `nao pod econom pod fat pod reputaca pod colater ter gent poder mao...` |

### Comentário emoji-only

| Versão             | Texto |
|--------------------|-------|
| Original           | `💛💛💛` |
| `text_clean`       | `:coração_amarelo::coração_amarelo::coração_amarelo:` |
| `text_preprocessed`| _(vazio)_ |

---

## 8. Arquivos gerados

```
data/processed/
├── cleaned_comments.csv
│   Colunas: videoId, video_title, commentId, publishedAt (datetime),
│            text, likeCount, channel_type, text_length, is_substantive
│   Linhas: 14.760
│
├── cleaned_metadata.csv
│   Colunas: videoId, title, description, channelId, channelTitle,
│            publishedAt (datetime), viewCount, likeCount, commentCount,
│            channel_type
│   Linhas: 40
│
└── preprocessed_comments.csv
    Colunas: todas de cleaned_comments.csv +
             text_clean (para BERTimbau)
             text_preprocessed (para TF-IDF)
    Linhas: 14.760
```

Todos os arquivos são salvos em **UTF-8 com BOM** (`utf-8-sig`), que garante
compatibilidade com Excel e ferramentas Windows sem problemas de encoding.

---

## 9. Como re-executar

Se novos dados forem coletados, basta sobrescrever `raw_comments.csv` e
`video_metadata.csv` e rodar os scripts na ordem:

```bash
# Na raiz do repositório:
python scripts/01_clean_data.py
python scripts/02_preprocess_text.py
```

**Dependências necessárias:**

```bash
pip install pandas nltk emoji
```

Os recursos do NLTK (stopwords, tokenizador, stemmer) são baixados
automaticamente na primeira execução do script 2.

---

*Documento gerado em 18/03/2026. Scripts em `scripts/`. Dados em `data/processed/`.*
