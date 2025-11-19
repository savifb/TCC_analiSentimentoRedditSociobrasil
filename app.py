import streamlit as st
import pandas as pd
import os
import plotly.express as px

st.set_page_config(page_title="Lab Opinion SocioBrasil", layout="wide")

# ----------------------------
# ESTILO GLOBAL (Tema Claro Gov.br)
# ----------------------------
st.markdown("""
<style>
    body, .main, .stApp { background-color: #FFFFFF !important; }
    section[data-testid="stSidebar"] { background-color: #E9F2FF; }
    .gov-header {
        background: linear-gradient(90deg, #1351B4, #0A3A83);
        padding: 22px;
        border-radius: 12px;
        text-align: center;
        color: white;
        margin-bottom: 25px;
    }
    .gov-card {
        background-color: #F2F7FF;
        border-left: 6px solid #1351B4;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0px 2px 4px rgba(0,0,0,0.08);
    }
    .gov-card h3 { color: #0F2A5F; font-weight: 600; margin-bottom: 4px; }
    .gov-card h1 { color: #1351B4; font-size: 32px; margin: 0; }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# MAPA DOS ARQUIVOS
# ----------------------------
dataset_arquivo = {
    "Postagens Sobre O STF": "stf_posts_sentimentoDeVerdade.csv",
    "Comentários Sobre O STF": "stf_comentarios_sentimento.csv",
    "As Postagens Sobre O Auxílio Brasil/Bolsa Família": "dfpostsAB.csv",
    "Comentários Sobre O Auxílio Brasil/Bolsa Família": "dfcomentariosAB.csv",
    "As Postagens Sobre A Vacinação Contra A Covid-19": "PostsVacinacaoSaude_final.csv",
    "Comentários Sobre A Vacinação Contra A Covid-19": "ComentariosVacinacaoSaude_final.csv",
}

# ----------------------------
# FUNÇÃO PARA CARREGAR DADOS SOB DEMANDA
# ----------------------------
@st.cache_data
def load_data(arquivo):
    base_path = "data"
    caminho = os.path.join(base_path, arquivo)
    sep = ',' if arquivo in {"stf_comentarios_sentimento.csv","dfpostsAB.csv","dfcomentariosAB.csv"} else ';'
    
    df = pd.read_csv(caminho, sep=sep, on_bad_lines='skip', engine='python')
    df.columns = df.columns.str.strip()
    if "Classe Sentimeto" in df.columns:
        df = df.rename(columns={"Classe Sentimeto": "Classe Sentimento"})
    df = df.drop(columns=["Unnamed: 0.1", "Unnamed: 0", 'Idioma', 'Subreddit', 'Link'], errors="ignore")
    return df

# ----------------------------
# HEADER
# ----------------------------
st.markdown("""
<div class="gov-header">
    <h1 style="margin-bottom: 0;">📊 SentimentLab SocioBrasil</h1>
    <p style="font-size:17px;margin-top:4px;">
        Plataforma de Análise de Sentimentos em Discussões do Reddit
    </p>
</div>
""", unsafe_allow_html=True)

# ----------------------------
# SIDEBAR
# ----------------------------
st.sidebar.title("📌 Navegação")
dataset_selecionado = st.sidebar.selectbox("Escolha o dataset", list(dataset_arquivo.keys()))

# Carrega apenas o dataset selecionado
df = load_data(dataset_arquivo[dataset_selecionado])

# ----------------------------
# DATAFRAME
# ----------------------------
st.write(f"### 📊 Dataset selecionado: **{dataset_selecionado}**")
st.dataframe(df, use_container_width=True)

# ----------------------------
# MÉTRICAS — CARDS GOV.BR
# ----------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""<div class="gov-card"><h3>Total de Registros</h3><h1>{len(df)}</h1></div>""", unsafe_allow_html=True)
with col2:
    positivos = df[df["Classe Sentimento"] == "POS"].shape[0]
    st.markdown(f"""<div class="gov-card"><h3>Positivos</h3><h1>{positivos}</h1></div>""", unsafe_allow_html=True)
with col3:
    negativos = df[df["Classe Sentimento"] == "NEG"].shape[0]
    st.markdown(f"""<div class="gov-card"><h3>Negativos</h3><h1>{negativos}</h1></div>""", unsafe_allow_html=True)
with col4:
    neutros = df[df["Classe Sentimento"] == "NEU"].shape[0]
    st.markdown(f"""<div class="gov-card"><h3>Neutros</h3><h1>{neutros}</h1></div>""", unsafe_allow_html=True)

# ----------------------------
# GRÁFICO
# ----------------------------
sent_counts = df["Classe Sentimento"].value_counts().reset_index()
sent_counts.columns = ["Sentimento", "Quantidade"]

fig = px.bar(
    sent_counts,
    x="Sentimento",
    y="Quantidade",
    title="Distribuição de Sentimentos",
    color="Sentimento",
    color_discrete_map={"POS": "#1351B4", "NEG": "#D22630", "NEU": "#F7C325"}
)
st.plotly_chart(fig, use_container_width=True)
