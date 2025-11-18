import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, roc_auc_score
)
from sklearn.preprocessing import label_binarize

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
def plot_roc(df, titulo):
    classes = ["NEG", "NEU", "POS"]

    y_true = df["rotulo"]
    y_score = df[["prob_NEG", "prob_NEU", "prob_POS"]]
    y_bin = label_binarize(y_true, classes=classes)

    aucs = {}

    plt.figure(figsize=(8, 6))
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score.iloc[:, i])
        auc_cls = roc_auc_score(y_bin[:, i], y_score.iloc[:, i])
        aucs[cls] = auc_cls
        plt.plot(fpr, tpr, label=f"{cls} – AUC {auc_cls:.2f}")

    plt.plot([0, 1], [0, 1], "k--")
    plt.title(f"Curvas ROC – {titulo}")
    plt.xlabel("Falsos Positivos (FPR)")
    plt.ylabel("Verdadeiros Positivos (TPR)")
    plt.legend()
    st.pyplot(plt)

    return aucs


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

# =========================================
# 7. CÁLCULO DAS MÉTRICAS (FLASH-CARDS POR CLASSE)
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

import plotly.express as px

# Matriz de confusão
matriz = confusion_matrix(y_true, y_pred, labels=["NEG", "NEU", "POS"])
df_matriz = pd.DataFrame(matriz, 
                         index=["NEG", "NEU", "POS"], 
                         columns=["NEG", "NEU", "POS"])

st.subheader("🔲 Matriz de Confusão")

fig = px.imshow(
    df_matriz,
    text_auto=True,       # coloca os números dentro das células
    color_continuous_scale="Blues",
    aspect="auto",
)

fig.update_layout(
    font=dict(color='black', size=14, family='Times New Roman'),
    xaxis_title="Predito",
    yaxis_title="Verdadeiro",
    coloraxis_colorbar_title="Quantidade",
    margin=dict(l=50, r=50, t=50, b=50),
)
fig.update_xaxes(tickfont=dict(color="black", size=14))
fig.update_yaxes(tickfont=dict(color="black", size=14))

fig.update_xaxes(side="top")  # coloca o eixo predito em cima (mais padrão em papers)

st.plotly_chart(fig, use_container_width=True)


# Especificidade por classe (usa sua função)
espec = calcular_especificidade(matriz, ["NEG", "NEU", "POS"])

# Função utilitária para ler valores com fallback
def get_metric(report, cls, metric):
    try:
        return report[cls][metric]
    except Exception:
        return 0.0

# Layout: 3 colunas (um flash-card por classe)
st.markdown("### 📑 Métricas por Classe")
col_neg, col_neu, col_pos = st.columns(3)

# Formatação bonita (valores em % com 1 casa)
def fmt(v):
    return f"{v*1:.1f}" 

with col_neg:
    st.markdown("#### 🔴 NEGATIVO")
    p = get_metric(report_dict, "NEG", "precision")
    r = get_metric(report_dict, "NEG", "recall")
    f1 = get_metric(report_dict, "NEG", "f1-score")
    s = espec.get("NEG", 0.0)
    # pode usar st.metric para destaque
    st.metric("Precisão", fmt(p))
    st.metric("Recall (Sensibilidade)", fmt(r))
    st.metric("F1-Score", fmt(f1))
    st.metric("Especificidade", fmt(s))

with col_neu:
    st.markdown("#### ⚪ NEUTRO")
    p = get_metric(report_dict, "NEU", "precision")
    r = get_metric(report_dict, "NEU", "recall")
    f1 = get_metric(report_dict, "NEU", "f1-score")
    s = espec.get("NEU", 0.0)
    st.metric("Precisão", fmt(p))
    st.metric("Recall (Sensibilidade)", fmt(r))
    st.metric("F1-Score", fmt(f1))
    st.metric("Especificidade", fmt(s))

with col_pos:
    st.markdown("#### 🟢 POSITIVO")
    p = get_metric(report_dict, "POS", "precision")
    r = get_metric(report_dict, "POS", "recall")
    f1 = get_metric(report_dict, "POS", "f1-score")
    s = espec.get("POS", 0.0)
    st.metric("Precisão", fmt(p))
    st.metric("Recall (Sensibilidade)", fmt(r))
    st.metric("F1-Score", fmt(f1))
    st.metric("Especificidade", fmt(s))
# =========================================
# Curva ROC
st.subheader("📈 Curvas ROC e AUC")
auc_resultados = plot_roc(df, opcao)


