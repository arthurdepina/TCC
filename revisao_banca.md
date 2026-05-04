# Revisão de Banca — TCC

**Título:** Análise comparativa do impacto de conteúdos sobre saúde mental no YouTube por meio de comentários: profissionais vs. não profissionais

**Data da avaliação:** 2026-05-04

---

## Nota Final: 6,1 / 10

| Critério | Peso | Nota |
|---|---|---|
| Formulação do problema e objetivos | 15% | 8,0 |
| Referencial teórico | 15% | 5,5 |
| Metodologia | 30% | 5,0 |
| Resultados e análise | 20% | 6,5 |
| Conclusões e limitações | 10% | 7,5 |
| Redação e formatação | 10% | 7,0 |

---

## Problema Metodológico Central — P3 (Intensidade Emocional)

> **Este é o erro mais grave do trabalho. Compromete toda uma pergunta de pesquisa.**

A P3 investiga se a *intensidade emocional* dos comentários difere entre canais profissionais e amadores. O problema está na definição operacional da variável, introduzida na Seção 3.8:

> *"a coluna `intensidade`, categorizada em três faixas: ALTA (confidence > 0,80), MÉDIA (0,60 ≤ confidence ≤ 0,80) e BAIXA (confidence < 0,60)"*

A coluna `intensidade` é simplesmente a **probabilidade de confiança do classificador XGBoost fatiada em três bandas**. Isso é a certeza do modelo sobre qual classe atribuir ao comentário — não é uma medida da intensidade emocional do texto.

**Por que isso é um erro conceitual:**

Um comentário pode ser fortemente emotivo ("Esse vídeo salvou minha vida, chorei do início ao fim!!!") e ao mesmo tempo apresentar features ambíguas que resultam em baixa confiança do classificador. O inverso também ocorre: um comentário emocionalmente neutro ("bom vídeo") pode ser classificado com alta confiança simplesmente por ser uma sentença curta e não ambígua. A confiança do modelo reflete a *separabilidade das features no espaço de decisão*, não a intensidade emocional do conteúdo.

**Impacto:** A afirmação *"canais amadores apresentam maior concentração de reações de alta intensidade (84,10% versus 73,95%)"* não pode ser sustentada com essa operacionalização. O que se demonstrou é que o XGBoost classifica comentários de canais amadores com maior certeza — o que pode refletir apenas que esses comentários são linguisticamente mais típicos de suas classes, não que são emocionalmente mais intensos.

**Caminhos de correção:**

1. **Renomear e reformular (mais rápido):** Chamar a variável de "certeza classificatória" e reescrever P3 como *"a certeza do classificador difere entre os grupos?"*. Não exige mudança no código.
2. **Medida léxica real (mais robusto):** Calcular um score de intensidade diretamente do texto — contagem de amplificadores ("muito", "extremamente", "demais"), sinais de pontuação exclamativos, uso de maiúsculas, repetição de caracteres ("nãooo"). Exige um novo script, mas tornaria P3 metodologicamente defensável.

---

## Outros Problemas Metodológicos

### Referência do BERTimbau incorreta (`@souza2020`)

A referência mais citada do trabalho contém múltiplos erros factuais:

- **Autor incorreto:** listado como `Fábio Capuano de Souza`; o nome correto é **Fábio Souza**
- **Co-autores ausentes:** Rodrigo Nogueira e Roberto Lotufo não constam
- **Venue incorreto:** o paper foi publicado no **BRACIS 2020** (Brazilian Symposium on Artificial Intelligence), não como documento avulso da USP
- **URL contraditória:** aponta para o repositório da **UNICAMP**, mas a instituição declarada é **USP**
- **Tipo incorreto:** está como `@misc` com `howpublished = {Universidade de São Paulo}`; deveria ser `@inproceedings`

Qualquer avaliador que conheça o BERTimbau identificará esse erro em 30 segundos.

### Estratégia half-split com viés de ordenação

A divisão treinamento/avaliação foi feita sequencialmente pela ordem de coleta. O YouTube retorna *top comments* ordenados por engajamento — os primeiros comentários coletados são os de maior engajamento. Treinar na primeira metade (mais engajamento) e avaliar na segunda (menos engajamento) introduz um viés sistemático não discutido. Uma divisão estratificada aleatória seria mais defensável.

### Ausência de medida de tamanho de efeito

Para P2, o χ² = 1.887,33 com n = 16.719. Com amostras grandes, diferenças triviais atingem significância estatística. O trabalho não reporta o **V de Cramér** nem o **w de Cohen** em nenhuma das perguntas, o que impede avaliar se as diferenças encontradas são substantivas ou apenas artefatos do tamanho amostral.

### P4 — Análise sobre 7,8% dos dados

92,2% dos comentários resultam em NEUTRA pela heurística léxica. A análise de P4 é restrita a 1.311 casos. Os resultados são apresentados com o mesmo peso retórico das demais perguntas, sem sinalizar adequadamente que as conclusões se aplicam apenas a um subconjunto muito pequeno e possivelmente não representativo do corpus.

### Concordância inter-anotadores não apresentada

600 comentários foram revisados manualmente por dois revisores. A concordância entre eles nunca é calculada nem apresentada na Seção de Resultados. Isso é uma lacuna na validação do *ground truth*.

### Variáveis de confundimento ignoradas

A comparação profissional vs. amador não controla tamanho do canal, recência dos vídeos, especificidade temática nem estilo de produção. Dois canais com 50k e 5M de inscritos têm dinâmicas de comentários incomparáveis sem estratificação.

---

## Problemas nas Referências Bibliográficas

| Referência | Problema |
|---|---|
| `@souza2020` | Autor errado, dois co-autores faltando, venue incorreto, URL contraditória, tipo errado |
| `@afonso2019` | Faltam volume, número e páginas para o artigo de revista |
| `@wang2023` | `author = {Wang, Ying and others}` — todos os autores devem ser listados na entrada `.bib` |
| `@bojanowski2016` | Publicado na TACL em 2017; citar como preprint de 2016 é impreciso |
| `@souza2021` | Citado 7+ vezes, incluindo contextos em que não é a referência mais adequada |

---

## Pontos Positivos

- Corpus expressivo: 42.372 comentários de 79 vídeos
- Pipeline reprodutível com 9 scripts documentados
- Escolha adequada de BERTimbau para português brasileiro
- Testes estatísticos (qui-quadrado e Kappa) aplicados corretamente
- Validação cruzada de 5 folds complementando o half-split
- Conclusão honesta com reconhecimento das limitações
- Resumo e abstract bem elaborados

---

## Correções Prioritárias (por impacto na nota)

1. Corrigir `@souza2020` — autores, co-autores, tipo `@inproceedings`, venue BRACIS 2020
2. Reescrever a operacionalização de P3 — renomear `intensidade` para `certeza_classificatoria` ou implementar score léxico real
3. Reportar V de Cramér para P2, P3 e P6
4. Apresentar concordância inter-anotadores da revisão humana na Seção 4
5. Corrigir `@afonso2019` com dados bibliográficos completos
