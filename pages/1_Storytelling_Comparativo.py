import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="Resultados Gerais", layout="wide")

# ================================
# ESTILOS GOV.BR (MESMO DA HOME)
# ================================

st.markdown("""
<style>
    body, .main, .stApp { background-color: #FFFFFF !important; }

    section[data-testid="stSidebar"] {
        background-color: #E9F2FF;
    }

    .gov-header {
        background: linear-gradient(90deg, #1351B4, #0A3A83);
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        color: white;
        margin-bottom: 30px;
    }

    .gov-card {
        background-color: #F2F7FF;
        border-left: 6px solid #1351B4;
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0px 2px 4px rgba(0,0,0,0.07);
        margin-bottom: 20px;
    }

    .gov-card h3 { color: #0F2A5F; margin-bottom: 5px; }
    .gov-card h1 { color: #1351B4; font-size: 30px; margin: 0; }
</style>
""", unsafe_allow_html=True)

# ================================
# CARREGAMENTO DOS DATASETS
# ================================

@st.cache_data
def load_data():
    base_path = "data"

    arquivos = {
        "STF_Posts": "stf_posts_sentimentoDeVerdade.csv",
        "STF_Comentarios": "stf_comentarios_sentimento.csv",
        "AB_Posts": "dfpostsAB.csv",
        "AB_Comentarios": "dfcomentariosAB.csv",
        "Vacina_Posts": "PostsVacinacaoSaude_final.csv",
        "Vacina_Comentarios": "ComentariosVacinacaoSaude_final.csv",
    }

    sep_virgula = {"stf_comentarios_sentimento.csv", "dfpostsAB.csv", "dfcomentariosAB.csv"}

    dfs = {}
    for nome, arq in arquivos.items():
        path = os.path.join(base_path, arq)
        sep = "," if arq in sep_virgula else ";"

        df = pd.read_csv(path, sep=sep, on_bad_lines='skip', engine='python')
        df.columns = df.columns.str.strip()

        if "Classe Sentimeto" in df.columns:
            df = df.rename(columns={"Classe Sentimeto": "Classe Sentimento"})

        df = df.drop(columns=["Unnamed: 0", "Unnamed: 0.1", "Idioma", "Subreddit", "Link"], errors="ignore")

        dfs[nome] = df
    return dfs

dfs = load_data()

# ================================
# HEADER
# ================================

st.markdown("""
<div class="gov-header">
    <h1>📖 Storytelling – Comparação Multitemática</h1>
    <p>Analisando a evolução histórica e a polaridade dos debates no Reddit</p>
</div>
""", unsafe_allow_html=True)

# ================================
# SEÇÃO 1 — VISÃO GERAL
# ================================

st.subheader("📌 Visão Geral dos Dados por Tema")

tema_info = {
    "STF": ("#1351B4", dfs["STF_Posts"], dfs["STF_Comentarios"]),
    "Auxílio Brasil / Bolsa Família": ("#0A7ABF", dfs["AB_Posts"], dfs["AB_Comentarios"]),
    "Vacinação": ("#1B98E0", dfs["Vacina_Posts"], dfs["Vacina_Comentarios"])
}

col1, col2, col3 = st.columns(3)

for col, (tema, (cor, posts, coments)) in zip([col1, col2, col3], tema_info.items()):
    with col:
        st.markdown(
            f"""
            <div class="gov-card">
                <h3>{tema}</h3>
                <p><b>Postagens:</b> {len(posts)}</p>
                <p><b>Comentários:</b> {len(coments)}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

# ================================
# SEÇÃO 2 — EVOLUÇÃO TEMPORAL
# ================================

st.subheader("📈 Evolução Histórica das Postagens por Tema")

# Garantir coluna de data
def ensure_date(df):
    for col in df.columns:
        if "date" in col.lower() or "data" in col.lower():
            df[col] = pd.to_datetime(df[col], errors='coerce')
            return col
    return None

evolucao_posts = []

for tema, (_, posts, _) in tema_info.items():
    coluna_data = ensure_date(posts)
    if coluna_data:
        temp = posts.groupby(posts[coluna_data].dt.to_period("M")).size().reset_index()
        temp.columns = ["Mes", "Quantidade"]
        temp["Tema"] = tema
        evolucao_posts.append(temp)

if evolucao_posts:
    df_evo = pd.concat(evolucao_posts)
    df_evo["Mes"] = df_evo["Mes"].astype(str)

    fig = px.line(
        df_evo,
        x="Mes",
        y="Quantidade",
        color="Tema",
        title="Evolução Mensal das Postagens",
        markers=True
    )
    st.plotly_chart(fig, use_container_width=True)

# ================================
# SEÇÃO 3 — DISTRIBUIÇÃO DE SENTIMENTOS (POSTAGENS)
# ================================

st.subheader("💬 Distribuição de Sentimentos — POSTAGENS")

dist_posts = []

for tema, (_, posts, _) in tema_info.items():
    temp = posts["Classe Sentimento"].value_counts().reset_index()
    temp.columns = ["Sentimento", "Quantidade"]
    temp["Tema"] = tema
    dist_posts.append(temp)

df_dist_posts = pd.concat(dist_posts)

fig2 = px.bar(
    df_dist_posts,
    x="Tema",
    y="Quantidade",
    color="Sentimento",
    barmode="group",
    title="Sentimentos nas Postagens por Tema",
    color_discrete_map={"POS": "#1351B4", "NEG": "#D22630", "NEU": "#F7C325"}
)
st.plotly_chart(fig2, use_container_width=True)

# ================================
# SEÇÃO 4 — DISTRIBUIÇÃO DE SENTIMENTOS (COMENTÁRIOS)
# ================================

st.subheader("💬 Distribuição de Sentimentos — COMENTÁRIOS")

dist_com = []

for tema, (_, _, coments) in tema_info.items():
    temp = coments["Classe Sentimento"].value_counts().reset_index()
    temp.columns = ["Sentimento", "Quantidade"]
    temp["Tema"] = tema
    dist_com.append(temp)

df_dist_com = pd.concat(dist_com)

fig3 = px.bar(
    df_dist_com,
    x="Tema",
    y="Quantidade",
    color="Sentimento",
    barmode="group",
    title="Sentimentos nos Comentários por Tema",
    color_discrete_map={"POS": "#1351B4", "NEG": "#D22630", "NEU": "#F7C325"}
)
st.plotly_chart(fig3, use_container_width=True)
