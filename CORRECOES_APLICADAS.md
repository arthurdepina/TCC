# Correções Aplicadas

## 1. Referência do BERTimbau (`souza2020`) — referencias.bib

### Erros corrigidos

| Campo | Antes (errado) | Depois (correto) |
|---|---|---|
| Tipo | `@misc` | `@inproceedings` |
| Autores | Souza, Fábio Capuano de (apenas 1) | Souza, Fábio + Nogueira, Rodrigo + Lotufo, Roberto de Alencar |
| Venue | Universidade de São Paulo | BRACIS 2020 — 9th Brazilian Conference on Intelligent Systems (Springer, LNCS vol. 12319) |
| Páginas | — | 403--417 |
| DOI | — | `10.1007/978-3-030-61377-8_28` |
| URL | repositório UNICAMP | Springer (link direto ao capítulo) |

### Entrada corrigida

```bibtex
@inproceedings{souza2020,
  author    = {Souza, Fábio and Nogueira, Rodrigo and Lotufo, Roberto de Alencar},
  title     = {{BERTimbau}: Pretrained {BERT} Models for {Brazilian} {Portuguese}},
  booktitle = {Intelligent Systems: 9th Brazilian Conference, {BRACIS} 2020, Rio Grande, Brazil, October 20--23, 2020, Proceedings, Part {I}},
  year      = {2020},
  pages     = {403--417},
  publisher = {Springer},
  series    = {Lecture Notes in Computer Science},
  volume    = {12319},
  doi       = {10.1007/978-3-030-61377-8_28},
  url       = {https://link.springer.com/chapter/10.1007/978-3-030-61377-8_28}
}
```

---

## 2. Texto do artigo — concordância gramatical (artigo.md, seção 2.3)

**Erro:** Com a correção do BERTimbau para três autores, o trecho que dizia "O autor apresentou" ficou gramaticalmente inconsistente (singular para três autores).

**Correção aplicada:**

| Antes | Depois |
|---|---|
| `@souza2020 desenvolveu o BERTimbau` | `@souza2020 desenvolveram o BERTimbau` |
| `O autor apresentou duas versões` | `Os autores apresentaram duas versões` |

---

## 3. Referência `afonso2019` — campos complementados

| Campo | Antes | Depois |
|---|---|---|
| Autores | `Afonso, A. R. and Duque, C. G.` | `Afonso, Alexandre Ribeiro and Duque, Cláudio Gottschalg` |
| `number` | ausente | `3` |
| `doi` | ausente | `10.18225/ci.inf.v48i3.4315` |
| `address` | `Brasília` | removido (campo não padrão para `@article`) |

**Fonte:** página do artigo em https://revista.ibict.br/ciinf/article/view/4315. Páginas não publicadas (numeração por artigo na revista). **Recomenda-se validar manualmente.**

---

## 4. Referência `wang2023` → renomeada `tan2023` — erro grave corrigido

| Campo | Antes (errado) | Depois (correto) |
|---|---|---|
| Chave | `wang2023` | `tan2023` |
| Autores | `Wang, Ying and others` | `Tan, Kian Long and Lee, Chin Poo and Lim, Kian Ming` |
| `doi` | ausente | `10.3390/app13074550` |

O primeiro autor não é Wang — é Tan. A chave `@wang2023` estava errada por completo. Todas as 4 ocorrências de `@wang2023` no `artigo.md` foram substituídas por `@tan2023`.

**Fonte:** CrossRef API via `api.crossref.org/works/10.3390/app13074550` (fonte autoritativa para DOIs). **Recomenda-se validar manualmente acessando https://www.mdpi.com/2076-3417/13/7/4550**

---

## 5. Demais referências — verificadas e confirmadas corretas

| Chave | Referência | Status |
|---|---|---|
| `souza2021` | Sentiment Analysis on Brazilian Portuguese User Reviews — arXiv:2112.05459 | Correto |
| `ding2023` | Is GPT-3 a Good Data Annotator? — ACL 2023 | Correto |
| `tan2024` | LLMs for Data Annotation and Synthesis: A Survey — EMNLP 2024 | Correto |
| `chen2016` | XGBoost — KDD 2016 | Correto |
| `devlin2019` | BERT — arXiv:1810.04805 | Correto |
| `goodfellow2016` | Deep Learning — MIT Press | Correto |
| `pennington2014` | GloVe — EMNLP 2014 | Correto |
| `bojanowski2016` | FastText — arXiv:1607.04606 | Correto |
| `mikolov2013` | Word2Vec — arXiv:1301.3781 | Correto |
| `howard2018` | ULMFiT — arXiv:1801.06146 | Correto |
| `russell2021` | AI: A Modern Approach, 4ª ed. — Pearson | Correto |
| `jurafsky2023` | Speech and Language Processing, 3ª ed. | Correto |
| `li2021` | Survey on Text Classification — arXiv:2008.00364 | Correto |
| `anthropic2024` | The Claude 3 Model Family — Anthropic | Correto |
| `taboada2011` | Lexicon-based methods — Computational Linguistics 37(2) | Correto |
| `cohen1960` | A coefficient of agreement — Educ. and Psych. Measurement | Correto |
| `mchugh2013` | The chi-square test — Biochemia Medica | Correto |
| `landis1977` | The measurement of observer agreement — Biometrics | Correto |

