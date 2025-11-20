import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, roc_auc_score
)
from sklearn.preprocessing import label_binarize
import altair as alt

# =========================================
# 1. STORYTELLING
# =========================================
st.title("📊 Avaliação de Desempenho do Modelo – Análise de Sentimentos")

st.markdown("""
Esta página apresenta uma **avaliação completa do desempenho** do modelo de análise de sentimentos
(BERTweet.br), aplicada aos temas:

- **STF**
- **Auxílio Brasil / Bolsa Família**
- **Vacinação contra a Covid-19**

A avaliação utiliza **rótulos manuais** como verdade e compara com as predições automáticas, gerando:

✅ Acurácia  
✅ Precisão, Recall, F1-Score  
✅ Matriz de Confusão  
✅ Especificidade  
✅ Curvas ROC e AUC  

Selecione abaixo o conjunto de dados para visualizar sua análise.
""")

# =========================================
# 2. CARREGAMENTO DOS DATASETS
# =========================================
DATA_PATH = "data/"

FILES = {
    "STF – Posts": "amostraCompletaSTFPosts.csv",
    "STF – Comentários": "amostraCompletaSTFComentarios.csv",
    "Auxílio Brasil – Posts": "amostraCompletaABPosts.csv",
    "Auxílio Brasil – Comentários": "amostraCompletaABComentarios.csv",
    "Vacinação – Posts": "amostraCompletoVSPosts1.csv",
    "Vacinação – Comentários": "amostraCompletoVSComentarios1.csv",
}

@st.cache_data
def load_csv(path):
    return pd.read_csv(path, sep=";", encoding="utf-8")

# =========================================
# 3. PADRONIZAÇÃO
# =========================================
def padronizar(df):
    replace_map = {'neu': 'NEU', 'NEY': 'NEU', 'UNKNOWN': 'NEU', 'MEI': 'NEU', 
                   'NaN': 'NEU', 'BEG': 'NEG', 'BEY': 'NEU'}
    for col in ['Classe Sentimento', 'rotulo']:
        if col in df.columns:
            df[col] = df[col].fillna("NEU").astype(str)
            df[col] = df[col].replace(replace_map)
    return df

# =========================================
# 4. ESPECIFICIDADE
# =========================================
def calcular_especificidade(matriz, classes):
    espec = {}
    for i, cls in enumerate(classes):
        VN = matriz.sum() - (matriz[i, :].sum() + matriz[:, i].sum() - matriz[i, i])
        FP = matriz[:, i].sum() - matriz[i, i]
        espec[cls] = VN / (VN + FP)
    return espec

# =========================================
# 5. CURVAS ROC
# =========================================
def plot_roc_altair(df, titulo):
    classes = ["NEG", "NEU", "POS"]
    y_true = df["rotulo"]
    y_score = df[["prob_NEG", "prob_NEU", "prob_POS"]]
    y_bin = label_binarize(y_true, classes=classes)
    
    roc_list = []
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score.iloc[:, i])
        auc_cls = roc_auc_score(y_bin[:, i], y_score.iloc[:, i])
        for x, y in zip(fpr, tpr):
            roc_list.append({"FPR": x, "TPR": y, "Classe": f"{cls} – AUC {auc_cls:.2f}"})
    
    df_roc = pd.DataFrame(roc_list)
    chart = alt.Chart(df_roc).mark_line().encode(
        x="FPR",
        y="TPR",
        color="Classe",
        tooltip=["Classe", "FPR", "TPR"]
    ).properties(title=f"Curvas ROC – {titulo}")
    st.altair_chart(chart, use_container_width=True)

# =========================================
# 6. UI – SELEÇÃO DO DATASET
# =========================================
opcao = st.selectbox("Selecione o conjunto de dados:", list(FILES.keys()))

df = load_csv(DATA_PATH + FILES[opcao])
df = padronizar(df)

st.subheader("🔍 Distribuição dos Rótulos")
st.write(df["rotulo"].value_counts())

# =========================================
# 7. CÁLCULO DAS MÉTRICAS
# =========================================
from sklearn.metrics import precision_recall_fscore_support

y_true = df["rotulo"]
y_pred = df["Classe Sentimento"]

# Acurácia
acuracia = accuracy_score(y_true, y_pred)
st.subheader("🎯 Acurácia")
st.metric("Acurácia (%)", f"{acuracia*100:.2f}%")

# Relatório em dict para acesso individual
report_dict = classification_report(y_true, y_pred, output_dict=True)

# =========================================
# MATRIZ DE CONFUSÃO – Altair
# =========================================
matriz = confusion_matrix(y_true, y_pred, labels=["NEG", "NEU", "POS"])
df_matriz = pd.DataFrame(matriz, index=["NEG", "NEU", "POS"], columns=["NEG", "NEU", "POS"])
df_matriz_reset = df_matriz.reset_index().melt(id_vars="index")
df_matriz_reset.columns = ["Verdadeiro", "Predito", "Quantidade"]

st.subheader("🔲 Matriz de Confusão")
chart_cm = alt.Chart(df_matriz_reset).mark_rect().encode(
    x="Predito:O",
    y="Verdadeiro:O",
    color="Quantidade:Q",
    tooltip=["Verdadeiro", "Predito", "Quantidade"]
).properties(width=400, height=400)
st.altair_chart(chart_cm, use_container_width=True)

# Especificidade
espec = calcular_especificidade(matriz, ["NEG", "NEU", "POS"])

# Função utilitária
def get_metric(report, cls, metric):
    try:
        return report[cls][metric]
    except Exception:
        return 0.0

st.markdown("### 📑 Métricas por Classe")
col_neg, col_neu, col_pos = st.columns(3)
def fmt(v): return f"{v*1:.1f}" 

with col_neg:
    st.markdown("#### 🔴 NEGATIVO")
    st.metric("Precisão", fmt(get_metric(report_dict, "NEG", "precision")))
    st.metric("Recall (Sensibilidade)", fmt(get_metric(report_dict, "NEG", "recall")))
    st.metric("F1-Score", fmt(get_metric(report_dict, "NEG", "f1-score")))
    st.metric("Especificidade", fmt(espec.get("NEG", 0.0)))

with col_neu:
    st.markdown("#### ⚪ NEUTRO")
    st.metric("Precisão", fmt(get_metric(report_dict, "NEU", "precision")))
    st.metric("Recall (Sensibilidade)", fmt(get_metric(report_dict, "NEU", "recall")))
    st.metric("F1-Score", fmt(get_metric(report_dict, "NEU", "f1-score")))
    st.metric("Especificidade", fmt(espec.get("NEU", 0.0)))

with col_pos:
    st.markdown("#### 🟢 POSITIVO")
    st.metric("Precisão", fmt(get_metric(report_dict, "POS", "precision")))
    st.metric("Recall (Sensibilidade)", fmt(get_metric(report_dict, "POS", "recall")))
    st.metric("F1-Score", fmt(get_metric(report_dict, "POS", "f1-score")))
    st.metric("Especificidade", fmt(espec.get("POS", 0.0)))

# =========================================
# Curva ROC
st.subheader("📈 Curvas ROC e AUC")
plot_roc_altair(df, opcao)
