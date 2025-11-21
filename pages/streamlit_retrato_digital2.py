import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, roc_auc_score
)
from sklearn.preprocessing import label_binarize
import os

# ==================== CONFIGURAÇÃO DA PÁGINA ====================
st.set_page_config(
    page_title="Retrato Digital da Opinião Pública",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== CSS CUSTOMIZADO ====================
st.markdown("""
<style>
    /* Tema escuro moderno */
    .main {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: #e8e8e8;
    }
    
    /* Cabeçalho principal */
    .big-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(120deg, #00d4ff, #7c3aed, #ff006e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        animation: fadeInDown 1s ease-out;
    }
    
    .subtitle {
        text-align: center;
        font-size: 1.3rem;
        color: #a0a0a0;
        margin-bottom: 3rem;
        animation: fadeIn 1.5s ease-out;
    }
    
    /* Cards modernos */
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d1b4e 100%);
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 212, 255, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 212, 255, 0.25);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #00d4ff;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #a0a0a0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Seção de storytelling */
    .story-section {
        background: rgba(255, 255, 255, 0.05);
        border-left: 4px solid #00d4ff;
        padding: 1.5rem;
        margin: 1.5rem 0;
        border-radius: 8px;
    }
    
    .story-title {
        font-size: 1.8rem;
        color: #00d4ff;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    
    .story-text {
        font-size: 1.1rem;
        line-height: 1.8;
        color: #d0d0d0;
    }
    
    /* Highlight numbers */
    .highlight-number {
        color: #ff006e;
        font-weight: 700;
        font-size: 1.2em;
    }
    
    /* Animações */
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    /* Remover padding padrão */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Tabs customizadas */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px 10px 0 0;
        padding: 1rem 2rem;
        color: #a0a0a0;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ==================== FUNÇÕES DE CARREGAMENTO ====================
DATA_PATH = "data/"

ARQUIVOS_DATASET = {
    "STF": {
        "posts": "stf_posts_sentimentoDeVerdade.csv",
        "comentarios": "stf_comentarios_sentimento.csv",
        "posts_amostra": "amostraCompletaSTFPosts.csv",
        "comentarios_amostra": "amostraCompletaSTFComentarios.csv"
    },
    "Auxílio Brasil": {
        "posts": "dfpostsAB.csv",
        "comentarios": "dfcomentariosAB.csv",
        "posts_amostra": "amostraCompletaABPosts.csv",
        "comentarios_amostra": "amostraCompletaABComentarios.csv"
    },
    "Vacinação": {
        "posts": "PostsVacinacaoSaude_final.csv",
        "comentarios": "ComentariosVacinacaoSaude_final.csv",
        "posts_amostra": "amostraCompletoVSPosts1.csv",
        "comentarios_amostra": "amostraCompletoVSComentarios1.csv"
    }
}

@st.cache_data
def load_data(arquivo, tipo="completo"):
    caminho = os.path.join(DATA_PATH, arquivo)
    
    # Definir separador baseado no arquivo
    if arquivo in ["stf_comentarios_sentimento.csv", "dfpostsAB.csv", "dfcomentariosAB.csv"]:
        sep = ','
    else:
        sep = ';'
    
    try:
        df = pd.read_csv(caminho, sep=sep, on_bad_lines='skip', engine='python')
        df.columns = df.columns.str.strip()
        
        # Padronizar nome da coluna
        if "Classe Sentimeto" in df.columns:
            df = df.rename(columns={"Classe Sentimeto": "Classe Sentimento"})
        
        # Remover colunas desnecessárias
        df = df.drop(columns=["Unnamed: 0.1", "Unnamed: 0", 'Idioma', 'Subreddit', 'Link'], errors="ignore")
        
        # Padronizar valores de sentimento
        if tipo == "amostra" and "rotulo" in df.columns:
            replace_map = {'neu': 'NEU', 'NEY': 'NEU', 'UNKNOWN': 'NEU', 'MEI': 'NEU', 
                          'NaN': 'NEU', 'BEG': 'NEG', 'BEY': 'NEU'}
            df["rotulo"] = df["rotulo"].fillna("NEU").astype(str).replace(replace_map)
            if "Classe Sentimento" in df.columns:
                df["Classe Sentimento"] = df["Classe Sentimento"].fillna("NEU").astype(str).replace(replace_map)
        
        return df
    except Exception as e:
        st.error(f"Erro ao carregar {arquivo}: {e}")
        return pd.DataFrame()

@st.cache_data
def carregar_todos_dados():
    """Carrega todos os dados e agrega estatísticas"""
    dados_agregados = {
        'temas': [],
        'posts_total': [],
        'posts_neg': [],
        'posts_neu': [],
        'posts_pos': [],
        'comentarios_total': [],
        'comentarios_neg': [],
        'comentarios_neu': [],
        'comentarios_pos': []
    }
    
    for tema, arquivos in ARQUIVOS_DATASET.items():
        # Posts
        df_posts = load_data(arquivos['posts'], tipo="completo")
        # Comentários
        df_comentarios = load_data(arquivos['comentarios'], tipo="completo")
        
        dados_agregados['temas'].append(tema)
        dados_agregados['posts_total'].append(len(df_posts))
        dados_agregados['posts_neg'].append(len(df_posts[df_posts['Classe Sentimento'] == 'NEG']))
        dados_agregados['posts_neu'].append(len(df_posts[df_posts['Classe Sentimento'] == 'NEU']))
        dados_agregados['posts_pos'].append(len(df_posts[df_posts['Classe Sentimento'] == 'POS']))
        
        dados_agregados['comentarios_total'].append(len(df_comentarios))
        dados_agregados['comentarios_neg'].append(len(df_comentarios[df_comentarios['Classe Sentimento'] == 'NEG']))
        dados_agregados['comentarios_neu'].append(len(df_comentarios[df_comentarios['Classe Sentimento'] == 'NEU']))
        dados_agregados['comentarios_pos'].append(len(df_comentarios[df_comentarios['Classe Sentimento'] == 'POS']))
    
    return pd.DataFrame(dados_agregados)

def calcular_metricas_completas(tema):
    """Calcula todas as métricas de desempenho para um tema"""
    arquivos = ARQUIVOS_DATASET[tema]
    
    metricas = []
    
    for tipo, arquivo_key in [("Postagens", "posts_amostra"), ("Comentários", "comentarios_amostra")]:
        df = load_data(arquivos[arquivo_key], tipo="amostra")
        
        if df.empty or "rotulo" not in df.columns:
            continue
        
        y_true = df["rotulo"]
        y_pred = df["Classe Sentimento"]
        
        # Acurácia
        acuracia = accuracy_score(y_true, y_pred)
        
        # Métricas por classe
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        # Matriz de confusão e especificidade
        matriz = confusion_matrix(y_true, y_pred, labels=["NEG", "NEU", "POS"])
        
        for i, classe in enumerate(["NEG", "NEU", "POS"]):
            VN = matriz.sum() - (matriz[i, :].sum() + matriz[:, i].sum() - matriz[i, i])
            FP = matriz[:, i].sum() - matriz[i, i]
            especificidade = VN / (VN + FP) if (VN + FP) > 0 else 0
            
            # AUC se houver probabilidades
            auc = 0
            if all(col in df.columns for col in ["prob_NEG", "prob_NEU", "prob_POS"]):
                try:
                    y_bin = label_binarize(y_true, classes=["NEG", "NEU", "POS"])
                    y_score = df[["prob_NEG", "prob_NEU", "prob_POS"]]
                    auc = roc_auc_score(y_bin[:, i], y_score.iloc[:, i])
                except:
                    auc = 0
            
            metricas.append({
                'Tema': tema,
                'Tipo': tipo,
                'Classe': classe,
                'Precision': report.get(classe, {}).get('precision', 0),
                'Recall': report.get(classe, {}).get('recall', 0),
                'F1-Score': report.get(classe, {}).get('f1-score', 0),
                'Especificidade': especificidade,
                'AUC': auc,
                'Acurácia': acuracia
            })
    
    return pd.DataFrame(metricas)

# ==================== CARREGAR DADOS ====================
with st.spinner('Carregando dados... Isso pode levar alguns segundos.'):
    df_agregado = carregar_todos_dados()
    
    # Calcular métricas de todos os temas
    metricas_completas = pd.concat([
        calcular_metricas_completas("STF"),
        calcular_metricas_completas("Auxílio Brasil"),
        calcular_metricas_completas("Vacinação")
    ], ignore_index=True)

# ==================== PÁGINA PRINCIPAL ====================
st.markdown('<h1 class="big-title">O Retrato Digital da Opinião Pública Brasileira</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Uma jornada pelos debates que dividem o país • 2015-2025</p>', unsafe_allow_html=True)

# ==================== MÉTRICAS PRINCIPAIS ====================
st.markdown("### 📊 Visão Geral da Pesquisa")
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_posts = df_agregado['posts_total'].sum()
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total de Postagens</div>
        <div class="metric-value">{total_posts:,}</div>
        <div class="metric-label">Analisadas</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    total_comentarios = df_agregado['comentarios_total'].sum()
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total de Comentários</div>
        <div class="metric-value">{total_comentarios:,}</div>
        <div class="metric-label">Analisados</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    acuracia_media = metricas_completas['Acurácia'].mean()
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Acurácia Média</div>
        <div class="metric-value">{acuracia_media*100:.1f}%</div>
        <div class="metric-label">BERTweet.br</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    melhor_f1 = metricas_completas['F1-Score'].max()
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Melhor F1-Score</div>
        <div class="metric-value">{melhor_f1:.2f}</div>
        <div class="metric-label">Classe Neutra</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==================== NAVEGAÇÃO POR TABS ====================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📖 Storytelling", 
    "📈 Polaridades", 
    "🎯 Desempenho do Modelo",
    "📊 Comparativo",
    "🔬 Análise Detalhada"
])

# ==================== TAB 1: STORYTELLING ====================
with tab1:
    st.markdown('<div class="story-section">', unsafe_allow_html=True)
    st.markdown('<div class="story-title">Capítulo 1: O Ponto de Partida</div>', unsafe_allow_html=True)
    st.markdown(f'''
    <div class="story-text">
    Imagine entrar em uma sala repleta de brasileiros debatendo os temas mais polêmicos do país. 
    Você ouviria vozes exaltadas falando sobre o <span class="highlight-number">Supremo Tribunal Federal</span>, 
    discussões acaloradas sobre <span class="highlight-number">programas sociais</span>, e opiniões divergentes 
    sobre a <span class="highlight-number">vacinação contra a Covid-19</span>. 
    <br><br>
    Essa sala existe — e se chama <strong>Reddit</strong>. Analisamos <span class="highlight-number">{total_posts:,} postagens</span> 
    e <span class="highlight-number">{total_comentarios:,} comentários</span> para entender o que os brasileiros realmente pensam.
    </div>
    ''', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="story-section">', unsafe_allow_html=True)
        st.markdown('<div class="story-title">O STF: A Instituição que Não Convence</div>', unsafe_allow_html=True)
        stf_posts_neg = df_agregado[df_agregado['temas'] == 'STF']['posts_neg'].values[0]
        stf_com_neg = df_agregado[df_agregado['temas'] == 'STF']['comentarios_neg'].values[0]
        st.markdown(f'''
        <div class="story-text">
        De <strong>fevereiro/2015 a junho/2025</strong> — uma década inteira de conversas sobre o STF.
        <br><br>
        • <span class="highlight-number">{stf_posts_neg}</span> postagens negativas<br>
        • <span class="highlight-number">{stf_com_neg}</span> comentários negativos<br>
        • O STF enfrenta <strong>crise de legitimidade digital</strong><br>
        • Negatividade supera amplamente o apoio
        </div>
        ''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="story-section">', unsafe_allow_html=True)
        st.markdown('<div class="story-title">Auxílio Brasil: A Promessa Inacabada</div>', unsafe_allow_html=True)
        ab_com_neg = df_agregado[df_agregado['temas'] == 'Auxílio Brasil']['comentarios_neg'].values[0]
        ab_com_pos = df_agregado[df_agregado['temas'] == 'Auxílio Brasil']['comentarios_pos'].values[0]
        st.markdown(f'''
        <div class="story-text">
        <span class="highlight-number">{ab_com_neg} negativas</span> vs <span class="highlight-number">{ab_com_pos} positivas</span>
        <br><br>
        • Frustrações com <strong>burocracia e cadastro</strong><br>
        • Questionamentos sobre <strong>elegibilidade</strong><br>
        • Valor considerado <strong>insuficiente</strong><br>
        • Persistência histórica de negatividade
        </div>
        ''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="story-section">', unsafe_allow_html=True)
    st.markdown('<div class="story-title">Covid-19: A Vacina Aprovada, o Governo Reprovado</div>', unsafe_allow_html=True)
    vac_posts_neu = df_agregado[df_agregado['temas'] == 'Vacinação']['posts_neu'].values[0]
    vac_com_neg = df_agregado[df_agregado['temas'] == 'Vacinação']['comentarios_neg'].values[0]
    st.markdown(f'''
    <div class="story-text">
    Um paradoxo fascinante: <strong>apoiam a ciência, desconfiam do governo</strong>.
    <br><br>
    • <span class="highlight-number">{vac_posts_neu:,} postagens neutras</span> com dados e estatísticas<br>
    • <span class="highlight-number">{vac_com_neg:,} comentários negativos</span> sobre gestão governamental<br>
    • Distinção clara: não é a vacina em julgamento, é a <strong>gestão governamental</strong>
    </div>
    ''', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ==================== TAB 2: POLARIDADES ====================
with tab2:
    st.markdown("### 📊 Distribuição de Sentimentos por Tema")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📝 Postagens")
        posts_data = pd.DataFrame({
            'Tema': df_agregado['temas'].tolist() * 3,
            'Polaridade': ['Negativo'] * 3 + ['Neutro'] * 3 + ['Positivo'] * 3,
            'Quantidade': (df_agregado['posts_neg'].tolist() + 
                          df_agregado['posts_neu'].tolist() + 
                          df_agregado['posts_pos'].tolist())
        })
        
        chart_posts = alt.Chart(posts_data).mark_bar().encode(
            x=alt.X('Tema:N', title='Tema'),
            y=alt.Y('Quantidade:Q', title='Quantidade'),
            color=alt.Color('Polaridade:N', 
                          scale=alt.Scale(domain=['Negativo', 'Neutro', 'Positivo'],
                                        range=['#ff006e', '#00d4ff', '#00f5a0'])),
            xOffset='Polaridade:N',
            tooltip=['Tema', 'Polaridade', 'Quantidade']
        ).properties(height=400)
        
        st.altair_chart(chart_posts, use_container_width=True)
    
    with col2:
        st.markdown("#### 💬 Comentários")
        comments_data = pd.DataFrame({
            'Tema': df_agregado['temas'].tolist() * 3,
            'Polaridade': ['Negativo'] * 3 + ['Neutro'] * 3 + ['Positivo'] * 3,
            'Quantidade': (df_agregado['comentarios_neg'].tolist() + 
                          df_agregado['comentarios_neu'].tolist() + 
                          df_agregado['comentarios_pos'].tolist())
        })
        
        chart_comments = alt.Chart(comments_data).mark_bar().encode(
            x=alt.X('Tema:N', title='Tema'),
            y=alt.Y('Quantidade:Q', title='Quantidade'),
            color=alt.Color('Polaridade:N',
                          scale=alt.Scale(domain=['Negativo', 'Neutro', 'Positivo'],
                                        range=['#ff006e', '#00d4ff', '#00f5a0'])),
            xOffset='Polaridade:N',
            tooltip=['Tema', 'Polaridade', 'Quantidade']
        ).properties(height=400)
        
        st.altair_chart(chart_comments, use_container_width=True)
    
    # Proporções por tema
    st.markdown("### 🎯 Análise Proporcional por Tema")
    tema_sel = st.selectbox("Selecione o tema:", df_agregado['temas'].tolist())
    
    idx = df_agregado[df_agregado['temas'] == tema_sel].index[0]
    
    col1, col2 = st.columns(2)
    
    with col1:
        post_data = pd.DataFrame({
            'Polaridade': ['Negativo', 'Neutro', 'Positivo'],
            'Quantidade': [df_agregado.iloc[idx]['posts_neg'],
                          df_agregado.iloc[idx]['posts_neu'],
                          df_agregado.iloc[idx]['posts_pos']]
        })
        
        pie_posts = alt.Chart(post_data).mark_arc(innerRadius=50).encode(
            theta=alt.Theta('Quantidade:Q'),
            color=alt.Color('Polaridade:N',
                          scale=alt.Scale(domain=['Negativo', 'Neutro', 'Positivo'],
                                        range=['#ff006e', '#00d4ff', '#00f5a0'])),
            tooltip=['Polaridade', 'Quantidade']
        ).properties(height=350, title='Postagens')
        
        st.altair_chart(pie_posts, use_container_width=True)
    
    with col2:
        comment_data = pd.DataFrame({
            'Polaridade': ['Negativo', 'Neutro', 'Positivo'],
            'Quantidade': [df_agregado.iloc[idx]['comentarios_neg'],
                          df_agregado.iloc[idx]['comentarios_neu'],
                          df_agregado.iloc[idx]['comentarios_pos']]
        })
        
        pie_comments = alt.Chart(comment_data).mark_arc(innerRadius=50).encode(
            theta=alt.Theta('Quantidade:Q'),
            color=alt.Color('Polaridade:N',
                          scale=alt.Scale(domain=['Negativo', 'Neutro', 'Positivo'],
                                        range=['#ff006e', '#00d4ff', '#00f5a0'])),
            tooltip=['Polaridade', 'Quantidade']
        ).properties(height=350, title='Comentários')
        
        st.altair_chart(pie_comments, use_container_width=True)

# ==================== TAB 3: DESEMPENHO ====================
with tab3:
    st.markdown("### 🎯 Avaliação de Desempenho do Modelo BERTweet.br")
    
    # Seleção de tema e tipo
    col1, col2 = st.columns(2)
    with col1:
        tema_metrica = st.selectbox("Tema:", ["Todos"] + df_agregado['temas'].tolist(), key="tema_metric")
    with col2:
        tipo_metrica = st.selectbox("Tipo de Texto:", ["Todos", "Postagens", "Comentários"], key="tipo_metric")
    
    # Filtrar métricas
    metricas_filtradas = metricas_completas.copy()
    if tema_metrica != "Todos":
        metricas_filtradas = metricas_filtradas[metricas_filtradas['Tema'] == tema_metrica]
    if tipo_metrica != "Todos":
        metricas_filtradas = metricas_filtradas[metricas_filtradas['Tipo'] == tipo_metrica]
    
    # Criar label combinada
    metricas_filtradas['Label'] = metricas_filtradas['Tema'] + ' - ' + metricas_filtradas['Classe']
    
    # Gráfico de métricas
    metrics_long = metricas_filtradas.melt(
        id_vars=['Label'],
        value_vars=['Precision', 'Recall', 'F1-Score'],
        var_name='Métrica',
        value_name='Score'
    )
    
    chart_metrics = alt.Chart(metrics_long).mark_bar().encode(
        x=alt.X('Label:N', title='', axis=alt.Axis(labelAngle=-45)),
        y=alt.Y('Score:Q', title='Score', scale=alt.Scale(domain=[0, 1])),
        color=alt.Color('Métrica:N', scale=alt.Scale(scheme='category10')),
        xOffset='Métrica:N',
        tooltip=['Label', 'Métrica', alt.Tooltip('Score:Q', format='.3f')]
    ).properties(height=400, title='Métricas de Desempenho')
    
    st.altair_chart(chart_metrics, use_container_width=True)
    
    # Tabela de métricas
    st.markdown("#### 📋 Tabela Completa de Métricas")
    st.dataframe(
        metricas_filtradas[['Tema', 'Tipo', 'Classe', 'Precision', 'Recall', 'F1-Score', 'Especificidade', 'Acurácia']].style.format({
            'Precision': '{:.3f}',
            'Recall': '{:.3f}',
            'F1-Score': '{:.3f}',
            'Especificidade': '{:.3f}',
            'Acurácia': '{:.3f}'
        }).background_gradient(cmap='viridis', subset=['Precision', 'Recall', 'F1-Score']),
        use_container_width=True,
        height=400
    )
    
    # Matriz de confusão para tema selecionado
    if tema_metrica != "Todos":
        st.markdown(f"### 🔲 Matriz de Confusão - {tema_metrica}")
        
        arquivo_amostra = ARQUIVOS_DATASET[tema_metrica]['comentarios_amostra']
        df_conf = load_data(arquivo_amostra, tipo="amostra")
        
        if not df_conf.empty and "rotulo" in df_conf.columns:
            matriz = confusion_matrix(df_conf["rotulo"], df_conf["Classe Sentimento"], labels=["NEG", "NEU", "POS"])
            
            cm_data = []
            for i, row_name in enumerate(["NEG", "NEU", "POS"]):
                for j, col_name in enumerate(["NEG", "NEU", "POS"]):
                    cm_data.append({
                        'Verdadeiro': row_name,
                        'Predito': col_name,
                        'Quantidade': int(matriz[i, j])
                    })
            
            cm_df = pd.DataFrame(cm_data)
            
            heatmap = alt.Chart(cm_df).mark_rect().encode(
                x=alt.X('Predito:N', title='Predito'),
                y=alt.Y('Verdadeiro:N', title='Verdadeiro'),
                color=alt.Color('Quantidade:Q', scale=alt.Scale(scheme='blues')),
                tooltip=['Verdadeiro', 'Predito', 'Quantidade']
            ).properties(width=400, height=400, title='Matriz de Confusão')