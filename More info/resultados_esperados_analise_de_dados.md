Perguntas que queremos responder:
1 - Os comentários mais engajados (com mais likes) expressam sentimentos diferentes entre os dois grupos?
Resultado do script de análise de dados: Percentual de cada rotulo dentro do grupo de comentários com likeCount maior ou igual a 10 por grupo
Tabelas resultantes: 
Visão percentual:
Colunas: Profissional, Amador
Linhas: 
Positivo: Preenchido com o percentual de ocorrências “POSITIVO” na coluna label_xgb
Negativo: Preenchido com o percentual de ocorrências “NEGATIVO” na coluna label_xgb
Tangencial: Preenchido com o percentual de ocorrências “TANGENCIAL” na coluna label_xgb
Visão valor absoluto:
Colunas: Profissional, Amador
Linhas: 
Positivo: Preenchido com o valor absoluto de ocorrências “POSITIVO” na coluna label_xgb
Negativo: Preenchido com o valor absoluto de ocorrências “NEGATIVO” na coluna label_xgb
Tangencial: Preenchido com o valor absoluto de ocorrências “TANGENCIAL” na coluna label_xgb
Lembrando dos filtro de likeCount >= 10
Para as perguntas abaixo será necessário apenas um csv:
Perguntas:
2 - Conteúdos de profissionais geram maior proporção de comentários positivos?
6 - Conteúdos de profissionais geram mais relatos pessoais (comentários TANGENCIAIS) do que conteúdos de amadores?
Resultado do script de análise de dados: Percentual de cada rotulo na base inteira por grupo
Tabelas resultantes: 
Visão percentual:
Colunas: Profissional, Amador
Linhas: 
Positivo: Preenchido com o percentual de ocorrências “POSITIVO” na coluna label_xgb
Negativo: Preenchido com o percentual de ocorrências “NEGATIVO” na coluna label_xgb
Tangencial: Preenchido com o percentual de ocorrências “TANGENCIAL” na coluna label_xgb
Visão valor absoluto:
Colunas: Profissional, Amador
Linhas: 
Positivo: Preenchido com o valor absoluto de ocorrências “POSITIVO” na coluna label_xgb
Negativo: Preenchido com o valor absoluto de ocorrências “NEGATIVO” na coluna label_xgb  
Tangencial: Preenchido com o valor absoluto de ocorrências “TANGENCIAL” na coluna label_xgb
3 - A intensidade emocional dos comentários difere entre profissionais e amadores? ← Modelo
Não utilizar o rótulo TANGENCIAL, não faz sentido
Resultado do script de análise de dados: Percentual de cada intensidade na base inteira por grupo
Tabelas resultantes:
Serão 4 tabelas no final. Onde cada grupo(amador e profissional) terão 2 tabelas, uma com a visão percentual e outra com a visão de valor absoluto, conforme descrito abaixo:
Visão percentual:
Colunas:
Total: Base total de rótulos do tipo POSITIVO e NEGATIVO
Positivo: Base total de rótulos do tipo POSITIVO 
Negativo: Base total de rótulos do tipo NEGATIVO
Linhas: 
Alta intensidade: Preenchido com o percentual de ocorrências de “ALTA” na coluna intensidade por grupo (Total, Positivo, Negativo)
Média intensidade: Preenchido com o percentual de ocorrências de “MÉDIA” na coluna intensidade por grupo (Total, Positivo, Negativo)
Baixa intensidade: Preenchido com o percentual de ocorrências de “BAIXA” na coluna intensidade por grupo (Total, Positivo, Negativo)
Visão valor absoluto:
Colunas:
Total: Base total de rótulos do tipo POSITIVO e NEGATIVO
Positivo: Base total de rótulos do tipo POSITIVO 
Negativo: Base total de rótulos do tipo NEGATIVO
Linhas: 
Alta intensidade: Preenchido com o valor absoluto de ocorrências de “ALTA” na coluna intensidade por grupo (Total, Positivo, Negativo)
Média intensidade: Preenchido com o valor absoluto de ocorrências de “MÉDIA” na coluna intensidade por grupo (Total, Positivo, Negativo)
Baixa intensidade: Preenchido com o valor absoluto de ocorrências de “BAIXA” na coluna intensidade por grupo (Total, Positivo, Negativo)
Lembrando do filtro de label_xgb IN (“POSITIVO ”, “NEGATIVO”)
4 - Conteúdos produzidos por profissionais apresentam mais indícios de confiabilidade percebida do que conteúdos produzidos por criadores sem formação na área? 
Resultado do script de análise de dados: Percentual de ocorrências do tipo ALTA e BAIXA confianca_percebida na base por grupo
ALTA indica o comentário contém termos que indicam confiança no conteúdo — ex: "explicou muito bem", "comprovado", "especialista", "faz sentido", "ajudou"
BAIXA	indica o comentário contém termos que questionam ou negam a credibilidade — ex: "pseudociência", "charlatão", "mentira", "não funciona", "absurdo"
Tabelas resultantes: 
Visão percentual:
Colunas: Profissional, Amador
Linhas: 
Alta confiança percebida: Preenchido com o percentual de ocorrências “ALTA” na coluna confianca_percebida
Baixa confiança percebida: Preenchido com o percentual de ocorrências “BAIXA” na coluna confianca_percebida
Visão valor absoluto:
Colunas: Profissional, Amador
Linhas: 
Alta confiança percebida: Preenchido com o valor absoluto de ocorrências “ALTA” na coluna confianca_percebida
Baixa confiança percebida: Preenchido com o valor absoluto de ocorrências “BAIXA” na coluna confianca_percebida
Lembrando do filtro de confianca_percebida IN (“ALTA”, “BAIXA”)
5 - Comparação entre a categorização feita pela LLM (profissional) e uma feita com XGBoost
Resultado do script de análise de dados: Visão da coluna concorda(quantidade de ocorrências de True e False) + Matriz de Confusão + Cohen's Kappa
Descrição de cada item esperado no resultado:
1. Coluna concorda (visão geral)
Responde: "quanto eles concordam no total?"
True:  14.178 (84,8%)
False:  2.541 (15,2%)
2. Matriz de confusão (visão detalhada)
Responde: "onde exatamente eles discordam?"
XGB: POSITIVO	XGB: TANGENCIAL	XGB: NEGATIVO
LLM: POSITIVO	✅ concordam	❌ LLM viu positivo, XGB viu tangencial	❌
LLM: TANGENCIAL	❌	✅ concordam	❌
LLM: NEGATIVO	❌	❌	✅ concordam
Isso revela padrões de discordância — por exemplo, se o XGBoost sistematicamente classifica como TANGENCIAL o que a LLM classificou como POSITIVO.
3. Cohen's Kappa (visão estatística)
Responde: "a concordância é real ou poderia ser por acaso?"
Sem o Kappa, 84,8% de concordância parece alto — mas se o dataset tem 56% de POSITIVOS, dois classificadores aleatórios já concordariam bastante só por isso. O Kappa desconta esse efeito e dá o número real de concordância.
6 - Conteúdos de profissionais geram mais relatos pessoais (comentários TANGENCIAIS) do que conteúdos de amadores?
A hipótese seria: criadores com autoridade percebida (profissionais de saúde mental) ativam mais identificação pessoal no público. Se a proporção de TANGENCIAL for maior em canais profissionais, isso sugere que o conteúdo ressoa além do racional — provoca memória afetiva.
Resultado do script de análise de dados: Percentual de ocorrências do rótulo “TANGENCIAL” por grupo(Profissional, Amador)
Tabelas resultantes: 
Visão percentual:
Colunas: Profissional, Amador
Linhas: 
Percentual de ocorrências: Preenchido com o percentual de ocorrências de “TANGENCIAL” na coluna label_xgb
Visão valor absoluto:
Colunas: Profissional, Amador
Linhas: 
Valor absoluto de ocorrências: Preenchido com o valor absoluto de ocorrências de “TANGENCIAL” na coluna label_xgb

