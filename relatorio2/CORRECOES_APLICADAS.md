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

## 3. Demais referências — verificadas e confirmadas corretas

| Chave | Referência | Status |
|---|---|---|
| `afonso2019` | Análise de sentimentos em comentários do YouTube — Ciência da Informação | Correto |
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

---

## 4. Ponto de atenção — wang2023 (requer verificação manual)

A revisão levantou uma dúvida sobre os autores da referência `wang2023`. O bib registra `Wang, Ying and others` para o artigo "A survey of sentiment analysis: approaches, datasets, and future research" (Applied Sciences 13(7), 4550, 2023). **Recomenda-se acessar a URL abaixo e confirmar se o primeiro autor é de fato Wang Ying:**

`https://www.mdpi.com/2076-3417/13/7/4550`

Se os autores estiverem errados, o campo `author` do `wang2023` no bib precisará ser corrigido. As afirmações sustentadas por essa referência no artigo (generalização de técnicas de NLP para análise de sentimentos) são plausíveis e compatíveis com um survey da área, independentemente de quem sejam os autores exatos.
