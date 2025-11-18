
# Documentação Técnica — SentimentLab (Streamlit)

## 1. Identificação
- Nome do software: SentimentLab
- Autor/Titular: [COLOQUE SEU NOME AQUI]
- Versão: 0.1 (preliminar)

## 2. Objetivo
SentimentLab é uma aplicação web para visualização interativa de análise de sentimentos aplicada a dados do Reddit. Permite explorar polaridade, comparar postagens e comentários, gerar métricas e exportar resultados.

## 3. Funcionalidades
- Visualização de contagens por sentimento (posts e comentários)
- Séries temporais de sentimento (se houver timestamps)
- Comparação entre posts e comentários
- Rankings de termos por sentimento e por volume
- Geração de curvas ROC/AUC (quando probabilidades estiverem disponíveis)
- Exportação de datasets filtrados

## 4. Arquitetura
- Frontend/Server: Streamlit (Python)
- Módulos: preprocessing.py, plots.py, model_utils.py, export_utils.py
- Deploy: Docker ou Streamlit Cloud

## 5. Entradas / Saídas
- Entrada: CSV com colunas mínimas: text, type (post/comment), sentiment
- Saída: gráficos interativos e CSVs exportáveis

## 6. Proteção de dados
Dados pessoais devem ser anonimizados antes do upload. O projeto inclui instruções e scripts para remoção de identificadores.

## 7. Anexos
- Prints das telas (gerar quando app estiver rodando)
- Amostra de dataset (anônima)
- Trechos relevantes do código-fonte
