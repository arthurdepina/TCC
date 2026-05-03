---
title: "Análise comparativa do impacto de conteúdos sobre saúde mental no YouTube por meio de comentários: profissionais vs. não profissionais"
author:
  - "Arthur de Pina Balduino Leitão^1^"
  - "Carolina Toledo Ferraz^1,2^"
affiliation:
  - "^1^Faculdade de Computação e Informática (FCI), Universidade Presbiteriana Mackenzie – São Paulo, SP – Brasil"
  - "^2^Programa de pós-graduação em Computação Aplicada, Universidade Presbiteriana Mackenzie – São Paulo, SP – Brasil"
email: "arthurdepinabalduinoleitao@gmail.com"
date: "2026"
lang: pt-BR
bibliography: referencias.bib
csl: abnt.csl
---

## Resumo

*Escreva o resumo aqui. Deve conter de 150 a 250 palavras, descrevendo contexto, motivação, objetivo, metodologia e resultados.*

**Palavras-chave:** Inteligência Artificial; Processamento de Linguagem Natural; Análise de Sentimentos; Saúde Mental; YouTube; BERTimbau.

## Abstract

*Write the abstract here. Must have between 150 to 250 words.*

**Keywords:** Artificial Intelligence; Natural Language Processing; Sentiment Analysis; Mental Health; YouTube; BERTimbau.

# 1. Introdução

## 1.1) Contextualização do problema de pesquisa

A crescente presença de conteúdos digitais sobre saúde e bem-estar nas plataformas de vídeo, especialmente no YouTube, tem provocado mudanças significativas na forma como as pessoas buscam e recebem informações sobre saúde mental e qualidade de vida. O consumo de vídeos que abordam temas como ansiedade, depressão, autocuidado e equilíbrio emocional aumentou expressivamente nos últimos anos, tornando o YouTube um espaço de troca e influência emocional. No entanto, essa popularização também traz desafios, pois parte dos conteúdos sobre saúde mental é produzida por pessoas sem formação técnica, o que pode gerar interpretações equivocadas, reforçar estigmas ou induzir comportamentos inadequados.

Nesse cenário, compreender como o público reage emocionalmente a esse tipo de conteúdo torna-se essencial e, mais especificamente, se essa reação varia conforme o perfil do criador. Comentários postados pelos usuários constituem um registro espontâneo de percepções e emoções, funcionando como indicadores de como determinados vídeos impactam a audiência. Contudo, o grande volume de comentários torna inviável uma análise manual, o que justifica o uso de abordagens automatizadas baseadas em Inteligência Artificial (IA) e Processamento de Linguagem Natural (NLP). Essas técnicas permitem extrair e interpretar padrões de sentimento presentes nas interações digitais, oferecendo um meio eficiente para avaliar o impacto emocional de conteúdos sobre saúde mental [@wang2023; @souza2021].

## 1.2) Definição e delimitação do problema de pesquisa

O problema central deste estudo consiste em compreender de que forma os vídeos sobre saúde mental e bem-estar disponíveis no YouTube influenciam emocionalmente seus espectadores, em particular, se esse impacto difere entre conteúdos produzidos por profissionais de saúde credenciados e por criadores sem formação técnica na área. Embora existam avanços em análise de sentimentos e aprendizado de máquina, grande parte das pesquisas ainda se concentra em textos escritos em inglês e em contextos formais, como artigos, resenhas e redes profissionais. Isso cria uma lacuna significativa quanto à aplicação dessas técnicas a textos em português brasileiro, especialmente quando se trata de comentários informais e espontâneos produzidos em plataformas digitais [@souza2021; @afonso2019].

Desse modo, o presente trabalho delimita-se à análise de comentários em língua portuguesa extraídos do YouTube, relacionados a vídeos sobre saúde mental e bem-estar de 12 canais; seis operados por profissionais de saúde (como médicos e psicólogos) e seis por criadores sem formação técnica na área. Os comentários foram classificados em quatro categorias de sentimento: POSITIVO (reação emocional favorável ao vídeo), NEGATIVO (reação emocional desfavorável), TANGENCIAL (relatos pessoais ou opiniões sobre o tema, sem referência direta ao vídeo) e DESCARTÁVEL (ausência de sentimento relevante). Essa classificação permite compreender de forma mais precisa o impacto emocional e social dos vídeos voltados à saúde e ao autocuidado mental na comunidade brasileira online, diferenciando o efeito produzido por criadores profissionais e amadores.

## 1.3) Objetivos Geral e Específicos da pesquisa

O objetivo geral desta pesquisa é analisar comparativamente o impacto emocional de vídeos do YouTube relacionados à saúde e bem-estar produzidos por profissionais de saúde e por criadores amadores, por meio da classificação automatizada de sentimentos expressos nos comentários dos espectadores. A proposta busca compreender de que forma os comentários refletem o efeito emocional diferenciado causado pelos dois perfis de criadores, contribuindo para uma visão mais ampla sobre como conteúdos digitais podem influenciar percepções, sentimentos e comportamentos associados ao autocuidado e à conscientização sobre saúde mental.

De forma complementar ao objetivo geral, a pesquisa buscou atingir os seguintes objetivos específicos:

- Coletar e organizar um corpus de comentários em língua portuguesa provenientes de canais profissionais e amadores de saúde mental e bem-estar no YouTube;
- Classificar automaticamente os comentários em categorias de sentimento utilizando um *pipeline* que combina rotulação por modelo de linguagem de grande porte (*Large Language Model*, LLM) e um classificador supervisionado baseado em *embeddings* contextuais gerados pelo modelo *BERTimbau*;
- Comparar padrões de sentimento, intensidade emocional e confiança percebida entre comentários de canais profissionais e amadores;
- Avaliar a concordância entre a rotulação realizada pelo LLM e as predições do classificador supervisionado, verificando a viabilidade do *pipeline* proposto.

## 1.4) Contribuições da pesquisa para a academia e sociedade

A relevância desta pesquisa manifesta-se em duas dimensões complementares: acadêmica e social.

Do ponto de vista acadêmico, o estudo amplia a aplicação de técnicas de Processamento de Linguagem Natural e aprendizado profundo à língua portuguesa, em um contexto ainda pouco explorado — o das interações informais em plataformas de vídeo. A utilização do modelo *BERTimbau* [@souza2020], pré-treinado especificamente para o português brasileiro, para a geração de *embeddings* contextuais representa um avanço técnico relevante ao adaptar abordagens de ponta a textos espontâneos. Adicionalmente, a adoção de um LLM para rotulação em larga escala e a comparação sistemática entre dois perfis distintos de criadores introduzem uma perspectiva nova no campo da análise de sentimentos em plataformas de vídeo. Assim, a pesquisa contribui para o estado da arte em NLP aplicado à análise de sentimentos, servindo de referência para investigações futuras em ambientes digitais.

No aspecto social, os resultados deste estudo fornecem evidências quantitativas sobre como o consumo de conteúdos de saúde mental produzidos por diferentes perfis de criadores afeta emocionalmente o público. Esses insumos podem subsidiar profissionais da saúde, comunicadores e plataformas digitais no desenvolvimento de estratégias mais responsáveis de divulgação e mediação de informação. Ao permitir uma leitura automatizada das reações emocionais dos espectadores, o estudo contribui para o fortalecimento de práticas comunicacionais que valorizem o bem-estar psicológico, a empatia e a disseminação de conteúdos confiáveis sobre saúde mental.

# 2. Referencial Teórico

## 2.1) Inteligência Artificial e Aprendizado de Máquina

A Inteligência Artificial (IA) é o campo da ciência da computação dedicado à criação de sistemas capazes de realizar tarefas que, quando executadas por humanos, exigiriam inteligência, como raciocínio, aprendizado e tomada de decisão. Esses sistemas processam grandes volumes de dados e extraem padrões complexos de forma automática, permitindo a construção de aplicações capazes de adaptar seu comportamento a novos contextos [@russell2021; @goodfellow2016]. No contexto da classificação automática de textos, o subcampo do Aprendizado de Máquina (*Machine Learning*), que permite que computadores imitem a maneira como os seres humanos aprendem, provê os métodos para que algoritmos aprendam padrões a partir de dados rotulados e generalizem decisões para novos exemplos [@wang2023]. @souza2021 destacam que abordagens supervisionadas têm sido amplamente aplicadas à análise de sentimentos, por possibilitarem a identificação automática de polaridades a partir de bases de comentários rotulados.

Apesar dos avanços significativos no campo do aprendizado de máquina, observa-se que grande parte dos estudos em análise de sentimentos permanece concentrada em contextos genéricos e em língua inglesa. Ainda há uma lacuna na aplicação de modelos supervisionados a dados em português e, especialmente, a interações digitais relacionadas à saúde mental. O presente trabalho busca contribuir nesse sentido, aplicando técnicas de aprendizado supervisionado a dados reais do YouTube, voltados a conteúdos de saúde e bem-estar.

## 2.2) Processamento de Linguagem Natural (NLP)

O Processamento de Linguagem Natural (*Natural Language Processing*, NLP) é uma subárea da Inteligência Artificial dedicada a permitir que máquinas compreendam, processem e analisem a linguagem humana [@jurafsky2023]. O NLP combina linguística computacional com técnicas estatísticas e redes neurais, modelos inspirados no cérebro humano que aprendem padrões a partir de dados, possibilitando a extração de significados e a análise semântica de textos [@goodfellow2016]. Segundo @souza2021, essa área é central para o desenvolvimento de sistemas de análise de sentimentos, uma vez que lida diretamente com a extração de significados e emoções em textos curtos e informais, como comentários em redes sociais e plataformas de vídeo.

No contexto deste trabalho, o NLP representa o elo entre a comunicação humana e os algoritmos de aprendizado de máquina. É a partir do processamento linguístico que se torna possível quantificar e interpretar o impacto emocional que os vídeos sobre saúde mental exercem sobre a comunidade que os assiste.

Embora o NLP tenha alcançado avanços expressivos, os estudos ainda se concentram em ambientes formais de linguagem, com dados estruturados e padronizados. Falta, portanto, uma investigação mais profunda sobre a aplicação dessas técnicas em contextos de linguagem informal, como comentários do YouTube, que refletem o comportamento emocional espontâneo dos usuários. Este trabalho busca suprir essa lacuna ao empregar NLP na análise de discursos informais e emocionais, oferecendo uma visão mais próxima da experiência social real dos usuários.

## 2.3) Representação Textual e Modelos de Classificação em NLP

O Processamento de Linguagem Natural exige a transformação de textos em representações numéricas, de modo que possam ser interpretados por algoritmos de aprendizado de máquina. Essa representação é a base para a aplicação de modelos de classificação, que têm por objetivo atribuir rótulos ou categorias a novos textos com base em padrões aprendidos a partir de exemplos previamente rotulados [@li2021; @wang2023].

Inicialmente, o NLP utilizava representações estatísticas, como o *TF-IDF* (*Term Frequency–Inverse Document Frequency*), que converte textos em vetores ponderados conforme a frequência e a relevância das palavras no corpus. Essa técnica se mostrou simples, eficiente e amplamente utilizada em tarefas supervisionadas. No estudo de @souza2021, o *TF-IDF* foi empregado com diferentes classificadores e obteve resultados expressivos, especialmente quando combinado ao *LightGBM* e à Regressão Logística. Apesar de eficientes, métodos baseados apenas em frequência não capturam o significado semântico das palavras nem suas relações contextuais.

Para superar essas limitações, surgiram as representações distribuídas (*word embeddings*), que mapeiam palavras para vetores contínuos de alta dimensão, de forma que termos semanticamente semelhantes ocupem posições próximas no espaço vetorial [@bojanowski2016; @souza2021]. Os modelos Word2Vec [@mikolov2013], GloVe [@pennington2014] e FastText [@bojanowski2016] marcaram essa evolução.

O avanço seguinte foi impulsionado pelo aprendizado profundo (*deep learning*) e, em particular, pelo *ULMFiT* [@howard2018], que introduziu o conceito de aprendizado por transferência em NLP, permitindo que modelos pré-treinados fossem ajustados a tarefas específicas. Posteriormente, o surgimento do *BERT* (*Bidirectional Encoder Representations from Transformers*) redefiniu o campo: proposto por @devlin2019, o *BERT* introduziu *embeddings* contextuais, nos quais a representação de cada palavra varia de acordo com o contexto em que aparece, superando as limitações dos *embeddings* estáticos.

No contexto do português brasileiro, @souza2020 desenvolveu o *BERTimbau*, o primeiro modelo *BERT* pré-treinado especificamente para a língua portuguesa, utilizando o *Brazilian Web as Corpus* (*brWaC*). O autor apresentou duas versões — *BERTimbau Base* e *BERTimbau Large* — que superaram o desempenho do modelo multilíngue *BERT* em tarefas como inferência textual, similaridade semântica e reconhecimento de entidades nomeadas. O uso de um vocabulário construído a partir de dados do português e o treinamento prolongado aumentam significativamente a sensibilidade do modelo às nuances da língua. Por essa razão, o presente trabalho utiliza o *BERTimbau* para a geração de *embeddings* contextuais dos comentários, como feature extractor com pesos congelados.

## 2.4) Análise de Sentimentos em Plataformas Digitais

A aplicação prática de técnicas de análise de sentimentos em contextos digitais é de grande relevância para compreender a recepção de conteúdos na internet. @afonso2019 analisaram comentários de vídeos do YouTube utilizando *SVM* (*Support Vector Machine*) com vetorização *TF-IDF* e demonstraram que a acurácia do modelo depende diretamente da uniformidade temática do corpus. Quando os comentários se referiam ao mesmo objeto (ou "entidade dominante"), o modelo alcançou 81% de acurácia e F1-score de 0,806, superando os experimentos mais amplos com múltiplos tópicos.

Os autores ressaltam que a linguagem informal, o uso de ironias e as variações linguísticas do português nas interações digitais representam desafios para a classificação automatizada de sentimentos. Esses achados reforçam a necessidade de modelos mais sensíveis ao contexto, como o *BERTimbau*, que é capaz de capturar nuances semânticas e emocionais. Embora os trabalhos existentes demonstrem o potencial das técnicas de aprendizado supervisionado em comentários do YouTube, ainda são raros os estudos que associam a análise de sentimentos ao impacto psicológico dos conteúdos de saúde mental e que comparam sistematicamente diferentes perfis de criadores. Dessa forma, este trabalho diferencia-se por propor uma abordagem que alia tecnologia e bem-estar digital.

## 2.5) NLP Aplicado à Saúde Mental e Bem-Estar

O uso de modelos de linguagem para compreender aspectos de saúde mental vem crescendo nos últimos anos. A análise de comentários em plataformas como o YouTube pode oferecer indicadores valiosos sobre o impacto psicológico e emocional dos conteúdos voltados ao bem-estar e à saúde. Ao aplicar técnicas de análise de sentimentos a esses dados, é possível identificar padrões de reação do público, revelando se os vídeos provocam percepções positivas, negativas ou neutras.

Essa abordagem se alinha ao que @souza2021 apontam como tendência: a utilização de modelos de aprendizado de máquina para interpretar interações humanas em larga escala, auxiliando na construção de ferramentas de monitoramento social e emocional. A incorporação de modelos de última geração, como o *BERTimbau* [@souza2020], permite elevar o nível de precisão dessas análises, contribuindo para que os resultados sirvam como suporte à conscientização sobre o impacto emocional do consumo de conteúdo digital relacionado à saúde mental. Apesar do avanço de modelos como o *BERTimbau*, observa-se uma escassez de pesquisas que apliquem essas arquiteturas ao domínio específico da saúde mental e ao ambiente multimodal do YouTube, lacuna que o presente estudo busca preencher.

## 2.6) Modelos de Linguagem de Grande Porte como Anotadores de Dados

A construção de bases de dados rotuladas é uma etapa fundamental em tarefas de aprendizado supervisionado, porém historicamente custosa: a anotação manual exige tempo, equipes de revisores especializados e processos de verificação de concordância entre anotadores. @tan2024 apresentam uma revisão abrangente do uso de Modelos de Linguagem de Grande Porte (*Large Language Models*, LLMs) como alternativa para a geração e síntese de anotações, demonstrando que LLMs podem automatizar tarefas de rotulagem com custo e tempo significativamente menores do que a anotação humana tradicional.

@ding2023 avaliaram sistematicamente o desempenho do GPT-3 como anotador em diversas tarefas de NLP e concluíram que o modelo pode produzir anotações de qualidade comparável à de anotadores humanos em tarefas de classificação de sentimentos, especialmente quando instruído com diretrizes claras e exemplos de referência (*few-shot prompting*). Esses resultados sustentam a viabilidade do uso de LLMs para rotulagem em larga escala, desde que a qualidade das anotações seja monitorada.

Neste trabalho, utilizou-se o modelo Claude Haiku, pertencente à família Claude 3 da Anthropic [@anthropic2024], para a rotulação automática dos 42.372 comentários coletados. A escolha do Claude Haiku se justifica por seu custo operacional reduzido, velocidade de processamento e desempenho competitivo em tarefas de compreensão e classificação de texto em português. Para garantir a qualidade das anotações, adotou-se um *prompt* detalhado com exemplos de referência e uma amostra estratificada de 600 comentários foi separada para revisão humana, permitindo estimar a confiabilidade da rotulação automática.

# 3. Metodologia

Esta seção descreve o desenvolvimento do trabalho, apresentando as etapas executadas desde a coleta dos dados até a geração dos *embeddings* contextuais. A pesquisa adota uma abordagem mista (quali-quanti): o aspecto quantitativo manifesta-se na mensuração de métricas de desempenho do modelo e na classificação das categorias de sentimento dos comentários; já o aspecto qualitativo evidencia-se na interpretação contextual dos resultados e na análise das reações do público diante do conteúdo assistido.

## 3.1) Visão Geral do Pipeline

O pipeline metodológico é composto por cinco etapas sequenciais, implementadas em scripts Python independentes e documentadas em código aberto:

1. **Coleta de dados** — extração de comentários e metadados via YouTube Data API v3;
2. **Limpeza e pré-processamento** — normalização do corpus e criação de duas versões textuais para diferentes modelos;
3. **Rotulação automática** — classificação de todos os comentários em quatro categorias de sentimento por meio de um LLM;
4. **Amostragem para revisão humana** — seleção estratificada de 600 comentários para validação das anotações;
5. **Geração de *embeddings*** — transformação dos comentários em vetores contextuais de 768 dimensões utilizando o *BERTimbau*.

As etapas subsequentes — treinamento do classificador supervisionado, avaliação do modelo e análise comparativa entre os grupos — são descritas na sequência da metodologia.

## 3.2) Seleção dos Canais e Critérios de Classificação

A seleção dos canais foi orientada pela distinção central desta pesquisa: comparar o impacto emocional de conteúdos produzidos por profissionais de saúde com o de conteúdos produzidos por criadores sem formação técnica na área. Foram selecionados 12 canais brasileiros do YouTube com foco em saúde mental e bem-estar, divididos em dois grupos:

**Canais profissionais** (criadores com formação acadêmica reconhecida na área de saúde):
Prazer, Karnal; Rossandro Klinjey; PodPeople – Ana Beatriz Barbosa; Augusto Cury; Minutos Psíquicos; Casa do Saber.

**Canais amadores** (criadores sem formação técnica na área de saúde):
ellora; JoutJout Prazer; Ludoviajante; Juliana Goes; Fred Elboni; Obvious.

O critério de classificação como "profissional" baseou-se na presença de formação acadêmica declarada na área de saúde (medicina, psicologia ou áreas correlatas) e na produção de conteúdo predominantemente técnico. Os canais amadores foram selecionados por sua relevância temática e audiência expressiva, sem vínculo com formação profissional na área de saúde. Ambos os grupos foram equilibrados em número de canais (seis cada) para permitir comparações proporcionais.

## 3.3) Coleta de Dados

A coleta de dados foi realizada de forma automatizada por meio de scripts em Python integrados à YouTube Data API v3, serviço oficial disponibilizado pelo Google que permite o acesso a informações públicas da plataforma. O processo de coleta foi guiado por dois parâmetros principais: até cinco vídeos por canal — selecionados entre os mais recentes com ao menos 650 comentários — e até 650 comentários por vídeo, coletados na ordem padrão retornada pela API (*top comments*).

Para cada comentário foram extraídos: identificador único (`commentId`), texto original, data de publicação, número de curtidas (`likeCount`), identificador do vídeo de origem e o tipo de canal (`channel_type`: `profissional` ou `amador`). Os metadados dos vídeos — título, descrição, contagem de visualizações, curtidas e comentários — foram coletados separadamente e armazenados em um arquivo auxiliar. Ao final da coleta, o corpus resultou em **79 vídeos** e **42.372 comentários**, com distribuição equilibrada entre os grupos: 50,4% de canais profissionais e 49,6% de canais amadores. Os dados foram armazenados em formato CSV (*Comma-Separated Values*), garantindo portabilidade e facilidade de tratamento nas etapas subsequentes. A coleta limitou-se a comentários de nível superior (*top-level comments*), excluindo respostas a comentários, de forma a padronizar a unidade de análise. Nenhuma informação pessoal identificável dos usuários foi retida, em conformidade com as normas éticas e de privacidade definidas pela plataforma e pela Lei Geral de Proteção de Dados (LGPD).

## 3.4) Limpeza e Pré-processamento dos Dados

Após a coleta, os dados passaram por duas etapas de tratamento sequenciais, implementadas nos scripts `01_clean_data.py` e `02_preprocess_text.py`.

**Limpeza estrutural (script 01):** foram removidos registros com texto nulo ou vazio, comentários duplicados (identificados pelo `commentId`), e registros com valores negativos para `likeCount`. Os *timestamps* foram convertidos para o formato UTC padronizado. Foram adicionadas colunas auxiliares: `text_length` (comprimento do texto em caracteres) e `is_substantive` (indicador booleano de que o comentário contém ao menos um caractere alfabético). Essas operações garantiram a integridade e a consistência do corpus antes das etapas de modelagem.

**Pré-processamento textual (script 02):** ao contrário de pipelines tradicionais de NLP que adotam uma única versão do texto, este trabalho criou duas representações textuais distintas, cada uma otimizada para um tipo diferente de modelo:

- `text_clean` — limpeza leve, adequada para o *BERTimbau*: remoção de URLs, menções (`@usuario`), *hashtags* (`#tópico`) e entidades HTML; conversão de emojis para descrição textual em português (por exemplo, 💛 → `:coração_amarelo:`); normalização de espaços. Maiúsculas, pontuação e acentos são preservados, pois o *BERTimbau* é um modelo *case-sensitive* que utiliza essas informações contextuais durante a codificação.

- `text_preprocessed` — pré-processamento completo, adequado para modelos tradicionais como *TF-IDF*: além das operações acima, conversão para minúsculas, remoção de acentos (normalização NFKD), remoção de caracteres não-linguísticos, tokenização, eliminação de *stopwords* em português (biblioteca NLTK) e *stemming* com o algoritmo RSLP (*Removedor de Sufixos da Língua Portuguesa*).

A razão para manter duas versões é técnica: modelos *Transformer* como o *BERTimbau* foram pré-treinados em texto real, com toda a sua riqueza contextual; normalizar o texto antes de passá-lo ao modelo destrói o contexto aprendido. Modelos baseados em frequência de palavras, por outro lado, beneficiam-se da normalização, pois ela reduz a esparsidade do vocabulário.

## 3.5) Rotulação com Modelo de Linguagem de Grande Porte

A rotulação do corpus foi realizada pelo modelo Claude Haiku [@anthropic2024], da família Claude 3 da Anthropic, via API, utilizando o script `03_llm_label.py`. A abordagem de usar um LLM como anotador automático em larga escala é respaldada pela literatura recente, que demonstra que modelos dessa classe podem produzir anotações de qualidade comparável à humana em tarefas de classificação de sentimentos [@ding2023; @tan2024].

**Schema de rotulação:** cada comentário foi classificado em uma de quatro categorias mutuamente exclusivas:

- **POSITIVO** — reação emocional favorável direcionada ao vídeo ou ao criador (gratidão, alívio após assistir, elogio ao conteúdo, identificação positiva com a abordagem);
- **NEGATIVO** — reação emocional desfavorável direcionada ao vídeo ou ao criador (crítica ao conteúdo, discordância da abordagem, avaliação do vídeo como inadequado ou perigoso);
- **TANGENCIAL** — comentário que tangencia o vídeo sem ser uma reação direta a ele (relato ou desabafo pessoal, opinião sobre o tema abordado — não sobre o vídeo em si, histórias de terceiros relacionadas ao assunto);
- **DESCARTÁVEL** — ausência de sentimento emocional relevante (perguntas factuais, *spam*, *timestamps*, emojis ambíguos isolados).

A regra central do *prompt* de sistema orientava o modelo a distinguir entre reações ao **vídeo** (classificadas como POSITIVO ou NEGATIVO) e reações ao **tema ou à própria vida** do comentarista (classificadas como TANGENCIAL). Para calibrar essa distinção, cada chamada à API incluía o título do vídeo antes do comentário, fornecendo o contexto necessário à classificação.

**Aspectos técnicos:** os comentários foram processados em lotes de 20 por chamada de API, com um *prompt* de sistema detalhado e oito exemplos de referência (*few-shot*). O script implementou um sistema de *checkpoint* automático, permitindo retomada sem reprocessamento em caso de interrupção, e um mecanismo de controle de custo com limite configurável (orçamento de R$ 40,00). A taxa de câmbio utilizada foi de R$ 5,80/USD, com tarifas de entrada de USD 0,80/MTok e saída de USD 4,00/MTok. O intervalo de 2,5 segundos entre lotes respeitou o limite de 50.000 tokens por minuto da API.

**Resultado:** todos os 42.372 comentários do corpus foram rotulados. A distribuição final foi: POSITIVO — 18.253 (43,1%); TANGENCIAL — 14.455 (34,1%); DESCARTÁVEL — 7.881 (18,6%); NEGATIVO — 1.783 (4,2%). A predominância de POSITIVO e TANGENCIAL e a raridade de NEGATIVO são consistentes com o comportamento típico de audiências em canais de saúde mental, onde comentários de suporte e relatos pessoais superam as críticas diretas.

## 3.6) Amostragem para Revisão Humana

Para estimar a confiabilidade da rotulação automática, o script `04_sample_for_review.py` gerou uma amostra estratificada de **600 comentários** para revisão manual, dividida igualitariamente entre dois revisores (300 cada). A estratificação priorizou a representação de categorias mais raras e potencialmente ambíguas — em especial NEGATIVO e comentários curtos — bem como comentários de alto engajamento (`likeCount` elevado), que têm maior peso analítico. A revisão humana permitirá calcular métricas de concordância entre revisores e entre humanos e LLM, validando a qualidade do *ground truth* utilizado na etapa de classificação supervisionada.

## 3.7) Geração de *Embeddings* com BERTimbau

A etapa final da fase de preparação de dados consistiu na transformação de cada comentário em um vetor numérico de alta dimensão, adequado para alimentar o classificador supervisionado. O script `05_generate_embeddings.py` utilizou o *BERTimbau* [@souza2020] — especificamente o modelo `neuralmind/bert-base-portuguese-cased` — como extrator de *features* com **pesos congelados** (sem *fine-tuning*). Para cada comentário, extraiu-se o vetor correspondente ao *token* `[CLS]` da última camada oculta do modelo, uma representação de 768 dimensões que condensa o significado semântico e contextual do texto completo.

Comentários rotulados como DESCARTÁVEL foram excluídos desta etapa, pois não carregam sentimento emocional relevante e sua inclusão introduziria ruído no treinamento do classificador. Os demais 34.491 comentários (POSITIVO, NEGATIVO e TANGENCIAL) foram processados com os seguintes parâmetros: comprimento máximo de 128 *tokens* (suficiente para cobrir mais de 99% dos comentários do corpus), tamanho de lote de 32, em GPU (NVIDIA RTX 4060 Laptop, 8 GB VRAM), com tempo total de execução de aproximadamente 2 minutos. O mapeamento de rótulos adotado foi: NEGATIVO → 0, TANGENCIAL → 1, POSITIVO → 2.

Os artefatos gerados foram: `embeddings.npy` — matriz de dimensões (34.491 × 768) em formato `float32`; `labels.npy` — vetor de inteiros com os rótulos correspondentes; e `embedding_meta.csv` — arquivo de metadados com `commentId`, rótulo LLM, tipo de canal, curtidas e título do vídeo.

---

> **[TODO — Seções 3.8 em diante]:** As etapas subsequentes do pipeline — validação cruzada do classificador XGBoost (script 06), treinamento e predição por divisão *half-split* (script 06b), construção do dataset final com heurística de confiança percebida (scripts 07 e 08) e análise estatística das seis perguntas de pesquisa (script 09) — serão descritas pela coautora na continuação desta seção.

# 4. Resultados e Discussão

*[A preencher]*

# 5. Conclusão

*[A preencher]*

# Referências Bibliográficas
