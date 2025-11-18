    # SentimentLab (Streamlit)


    SentimentLab — Dashboard interativo para visualização de análise de sentimentos (Reddit), construído com Streamlit.


    ## Como rodar localmente


    1. Crie e ative um ambiente virtual (recomendado)

```bash
python -m venv venv
# Linux/Mac
source venv/bin/activate
# Windows
venv\Scripts\activate
```

2. Instale dependências

```bash
pip install -r requirements.txt
```

3. Rode a aplicação

```bash
streamlit run app.py
```

Acesse: http://localhost:8501

## Estrutura do projeto

```
(sentimentlab_streamlit folder contents)
```

## Notas
- Substitua `data/sample_dataset.csv` por suas bases reais (posts e comentários). Certifique-se de anonimizar dados pessoais.
- Para ROC/AUC, inclua colunas `label_binary` (0/1) e `prob_positive` (probabilidade entre 0 e 1).
