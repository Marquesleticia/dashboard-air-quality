## 📊 Funcionalidades do Dashboard

O dashboard foi desenvolvido para análise exploratória e detecção de anomalias em dados de qualidade do ar utilizando o algoritmo **Isolation Forest**. A aplicação está organizada nas seguintes abas:

### 📋 Visão Geral

Apresenta uma visão geral do projeto e do conjunto de dados utilizado:

- Descrição do objetivo do projeto;
- Informações sobre o dataset Air Quality (UCI Machine Learning Repository);
- Período analisado (2004–2005);
- Descrição dos sensores e variáveis monitoradas;
- Indicadores globais (KPIs);
- Gráfico de pizza com a proporção entre registros **Normais** e **Anômalos**.

---

### 📊 Análise Exploratória

Disponibiliza análises estatísticas e visuais para compreensão dos dados:

- Histogramas de todas as variáveis numéricas;
- Boxplots segmentados por classificação (Normal × Anômalo);
- Heatmap de correlação entre variáveis;
- Séries temporais com destaque para os registros anômalos;
- Estatísticas descritivas das variáveis monitoradas.

---

### 🤖 Modelo

Apresenta detalhes sobre o modelo de detecção de anomalias utilizado:

- Explicação conceitual do algoritmo Isolation Forest;
- Parâmetros utilizados no treinamento;
- Variáveis (features) consideradas pelo modelo;
- Métricas de avaliação:
  - Silhouette Score;
  - Davies-Bouldin Index (DBI);
- Visualização PCA (Principal Component Analysis);
- Distribuição dos scores de anomalia.

---

### 🚨 Anomalias Detectadas

Permite analisar os registros identificados como anômalos:

- Quantidade de anomalias por mês;
- Quantidade de anomalias por hora do dia;
- Tabela detalhada dos registros anômalos;
- Ordenação por score de anomalia;
- Exportação dos resultados em formato CSV.

---

### 📝 Conclusão

Apresenta um resumo dos resultados obtidos:

- Principais descobertas da análise;
- Interpretação das anomalias detectadas;
- Limitações do estudo;
- Possíveis melhorias futuras;
- Referências bibliográficas utilizadas no projeto.