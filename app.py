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
st.markdown('<p class="subtitle">Uma analise sobre temas sociopolíticos brasileiros • 2015-2025</p>', unsafe_allow_html=True)

# RESULTADOS GERAIS - 1 SEÇÃO RESULTADOS E DISCUSSÕES - 

st.markdown("### Visão Geral da Pesquisa")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Período Analisado</div>
        <div class="metric-value">10 anos</div>
        <div class="metric-label">2015 - 2025</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Temas Estudados</div>
        <div class="metric-value">3</div>
        <div class="metric-label">STF • Auxílio • Vacina</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Acurácia do Modelo</div>
        <div class="metric-value">69-84%</div>
        <div class="metric-label">BERTweet.br</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Melhor F1-Score</div>
        <div class="metric-value">0.92</div>
        <div class="metric-label">Classe Neutra</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==================== NAVEGAÇÃO POR TABS ==================== 
tab1, tab4, tab2, tab3, tab5 = st.tabs([
    "📖 Visão Geral", # tab1 / 5
    "📅 Evolução Temporal", # tab5 / 1
    "📈 Polaridades",  # tab2 / 2 
    "🎯 Desempenho do Modelo", #tab3 / 4
    "🔬 Análise Detalhada", #tab4 / 5
    
])

# ==================== TAB 1: PRINCIPAIS CONCLUSÕES ====================
with tab1:

    import streamlit as st

    # Configuração da página
    st.set_page_config(page_title="Análise de Sentimentos", layout="wide")

    # --- ESTILO APRIMORADO ---
# --- ESTILO ADAPTATIVO - TEMA CLARO E ESCURO ---
    st.markdown("""
    <style>
        /* Importar fonte mais moderna */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        
        /* Reset e configurações gerais */
        .visao-geral-container {
            font-family: 'Inter', sans-serif;
        }
        
        /* Container principal */
        .visao-geral-container > div {
            padding: 2rem;
        }
        
        /* ============================================
           TEMA CLARO (padrão)
           ============================================ */
        
        /* Título principal - CLARO */
        .story-title {
            font-size: 32px; 
            font-weight: 700;
            margin-top: 40px;
            margin-bottom: 20px;
            color: #1a1a1a;
            border-left: 5px solid #2563eb;
            padding-left: 20px;
            line-height: 1.3;
        }
        
        /* Subtítulo - CLARO */
        .story-subtitle {
            font-size: 24px; 
            font-weight: 600;
            margin-top: 35px;
            margin-bottom: 15px;
            color: #2563eb;
            position: relative;
        }
        
        .story-subtitle::after {
            content: '';
            position: absolute;
            bottom: -8px;
            left: 0;
            width: 60px;
            height: 3px;
            background: linear-gradient(90deg, #2563eb, transparent);
        }
        
        /* Texto principal - CLARO */
        .story-text {
            font-size: 17px; 
            line-height: 1.8;
            color: #374151;
            margin-bottom: 25px;
            text-align: justify;
        }
        
        /* Destaque inline - CLARO */
        .highlight {
            font-weight: 600;
            color: #1e40af;
            background: linear-gradient(120deg, #dbeafe 0%, #dbeafe 100%);
            background-repeat: no-repeat;
            background-size: 100% 40%;
            background-position: 0 85%;
            padding: 2px 4px;
        }
        
        /* Lista - CLARO */
        .custom-list {
            margin: 20px 0;
            padding-left: 0;
        }
        
        .list-item {
            padding: 12px 20px;
            margin: 10px 0;
            background: #f8fafc;
            border-left: 4px solid #60a5fa;
            border-radius: 4px;
            font-size: 16px;
            color: #1e293b;
            transition: all 0.3s ease;
        }
        
        .list-item:hover {
            background: #eff6ff;
            transform: translateX(5px);
        }
        
        /* Card temático - CLARO */
        .theme-card {
            background: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            margin-bottom: 25px;
            border-top: 4px solid #2563eb;
            transition: all 0.3s ease;
        }
        
        .theme-card:hover {
            box-shadow: 0 4px 16px rgba(0,0,0,0.12);
            transform: translateY(-2px);
        }
        
        /* Separador - CLARO */
        .divider {
            height: 2px;
            background: linear-gradient(90deg, transparent, #e5e7eb, transparent);
            margin: 40px 0;
        }
        
        /* Badge - CLARO */
        .badge {
            display: inline-block;
            padding: 6px 14px;
            background: #dbeafe;
            color: #1e40af;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 600;
            margin-right: 10px;
            margin-bottom: 10px;
        }
        
        /* ============================================
           TEMA ESCURO (media query)
           ============================================ */
        
        @media (prefers-color-scheme: dark) {
            
            /* Título principal - ESCURO */
            .story-title {
                color: #e8e8e8;
                border-left-color: #00d4ff;
            }
            
            /* Subtítulo - ESCURO */
            .story-subtitle {
                color: #00d4ff;
            }
            
            .story-subtitle::after {
                background: linear-gradient(90deg, #00d4ff, transparent);
            }
            
            /* Texto principal - ESCURO */
            .story-text {
                color: #d0d0d0;
            }
            
            /* Destaque inline - ESCURO */
            .highlight {
                color: #00d4ff;
                background: linear-gradient(120deg, rgba(0, 212, 255, 0.2) 0%, rgba(0, 212, 255, 0.2) 100%);
            }
            
            /* Lista - ESCURO */
            .list-item {
                background: rgba(255, 255, 255, 0.05);
                border-left-color: #00d4ff;
                color: #d0d0d0;
            }
            
            .list-item:hover {
                background: rgba(0, 212, 255, 0.1);
            }
            
            /* Card temático - ESCURO */
            .theme-card {
                background: linear-gradient(135deg, #1e3a5f 0%, #2d1b4e 100%);
                box-shadow: 0 8px 32px rgba(0, 212, 255, 0.15);
                border-top-color: #00d4ff;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }
            
            .theme-card:hover {
                box-shadow: 0 12px 40px rgba(0, 212, 255, 0.25);
            }
            
            /* Separador - ESCURO */
            .divider {
                background: linear-gradient(90deg, transparent, rgba(0, 212, 255, 0.3), transparent);
            }
            
            /* Badge - ESCURO */
            .badge {
                background: rgba(0, 212, 255, 0.2);
                color: #00d4ff;
            }
        }
        
        /* Seção de destaque (mantém o mesmo em ambos os temas) */
        .highlight-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 30px;
            border-radius: 12px;
            color: white;
            margin: 30px 0;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        }
        
        .highlight-box .story-text {
            color: white;
        }
        
        /* Ajustes para colunas */
        [data-testid="column"] {
            padding: 0 15px;
        }
    </style>
    """, unsafe_allow_html=True)

    # ---------------- HEADER COM CONTEXTO ----------------
    st.markdown('<div class="story-title">Panorama Geral dos Resultados (2015–2025)</div>', unsafe_allow_html=True)
    st.markdown("""
    
    <div class="story-text">
    A década analisada revela um padrão emocional sólido: as discussões sociopolíticas no Reddit 
    brasileiro começam neutras, mas rapidamente se transformam em espaços de crítica. A neutralidade 
    domina a superfície; a negatividade domina a conversa. E o sentimento positivo permanece raro, 
    explícito apenas quando impossível de ignorar.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ---------------- DINÂMICA EMOCIONAL ----------------
    st.markdown('<div class="story-subtitle">💬 A Dinâmica da Conversa Pública</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="story-text">
    As postagens iniciais apresentam um tom informativo e distanciado, mas basta o diálogo começar 
    para que o padrão emocional mude. <span class="highlight">Comentários transformam informações neutras 
    em debates carregados de crítica</span>.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="custom-list">
        <div class="list-item">🔵 O neutro inicia a conversa</div>
        <div class="list-item">🔴 O negativo cresce exponencialmente</div>
        <div class="list-item">🟢 O positivo quase desaparece</div>
    </div>
    <div class="story-text">
    Esse movimento é consistente em todos os temas analisados.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ---------------- SEÇÃO DOS TEMAS ----------------
    st.markdown('<div class="story-title">🎯 As Emoções em Cada Tema</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    # --- STF ---
    with col1:
        st.markdown("""
        <div class="theme-card">
            <span class="badge">Poder Judiciário</span>
            <div class="story-subtitle">⚖️ STF: A Crítica Estrutural</div>
            <div class="story-text">
            No tema STF, <span class="highlight">a negatividade não é episódica — é estrutural</span>.  
            Postagens neutras quase sempre geram ondas de críticas diretas, revelando uma 
            dificuldade contínua da instituição em construir legitimidade discursiva no ambiente digital.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # --- AUXÍLIO BRASIL ---
    with col2:
        st.markdown("""
        <div class="theme-card">
            <span class="badge">Política Social</span>
            <div class="story-subtitle">💰 Auxílio Brasil: Frustração Repetida</div>
            <div class="story-text">
            A crítica domina as conversas sobre o Auxílio Brasil, reproduzindo um padrão histórico de 
            negatividade já registrado em estudos anteriores. Reclamações sobre burocracia, acesso, 
            valor insuficiente e politização do programa formam um conjunto de frustrações recorrentes.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # --- COVID (largura total) ---
    st.markdown("""
    <div class="theme-card">
        <span class="badge">Saúde Pública</span>
        <div class="story-subtitle">💉 Vacinação Covid-19: Ciência Aprovada, Gestão Reprovada</div>
        <div class="story-text">
        O tema traz o padrão mais paradoxal: confiança na ciência, reprovação à gestão governamental.  
        <span class="highlight">A crítica não recai sobre a vacina, mas sobre quem conduziu o processo</span>.  
        Enquanto a superfície permanece neutra, os comentários revelam insatisfação explícita 
        com políticas, atrasos e decisões governamentais.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ---------------- MÉTRICAS ----------------
    st.markdown('<div class="story-subtitle">📈 O Que as Métricas Contam Sobre o Modelo</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="story-text">
    O modelo apresenta desempenho consistente para identificar neutralidade e negatividade, 
    mas enfrenta dificuldades com sentimentos positivos — não por falha técnica, mas pela própria 
    natureza dos debates: <span class="highlight">o positivo é raro, sutil e muitas vezes irônico</span>.
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="list-item">
            <strong>🔵 Neutro</strong><br>
            Melhor desempenho, maior recall
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="list-item">
            <strong>🔴 Negativo</strong><br>
            Estabilidade consistente
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="list-item">
            <strong>🟢 Positivo</strong><br>
            Subdetecção pela raridade
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="story-text">
    As métricas confirmam o padrão emocional encontrado nos dados.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ---------------- FECHAMENTO ----------------
    st.markdown("""
    <div class="highlight-box">
        <div class="story-title" style="border: none; padding: 0; margin-top: 0; color: white;">
            ✨ Em Síntese
        </div>
        <div class="story-text">
        Os resultados revelam um ecossistema discursivo em que a informação abre o debate, 
        mas a crítica o define. A neutralidade domina a superfície; a negatividade domina o diálogo; 
        e o sentimento positivo, quando aparece, se torna exceção que confirma o padrão.
        </div>
    </div>
    """, unsafe_allow_html=True)




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
        
# ==================== TAB 3: DESEMPENHO DO MODELO ====================
# ==================== TAB 3: DESEMPENHO DO MODELO ====================
with tab3:
    st.markdown("### 🎯 Avaliação de Desempenho do Modelo BERTweet.br")
    
    st.markdown("""
    <div class="story-section">
        <div class="story-text">
        Esta seção apresenta uma <strong>avaliação completa do desempenho</strong> do modelo de análise de sentimentos
        BERTweet.br aplicado aos três temas: <strong>STF</strong>, <strong>Auxílio Brasil</strong> e <strong>Vacinação</strong>.
        <br><br>
        ✅ Acurácia, Precisão, Recall, F1-Score<br>
        ✅ Matriz de Confusão<br>
        ✅ Especificidade e AUC<br>
        ✅ Curvas ROC
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ==================== 1. SELEÇÃO E TABELA DE MÉTRICAS ====================
    st.markdown("#### 📊 Métricas de Desempenho por Tema")
    
    col1, col2 = st.columns(2)
    
    with col1:
        tema_sel_desempenho = st.selectbox(
            "Selecione o tema para análise:", 
            df_agregado['temas'].tolist(), 
            key='tema_desempenho'
        )
    
    with col2:
        tipo_sel_desempenho = st.selectbox(
            "Tipo de texto:",
            ["Todos", "Postagens", "Comentários"],
            key='tipo_desempenho'
        )
    
    # Filtrar métricas
    metricas_filtradas = metricas_completas[metricas_completas['Tema'] == tema_sel_desempenho]
    if tipo_sel_desempenho != "Todos":
        metricas_filtradas = metricas_filtradas[metricas_filtradas['Tipo'] == tipo_sel_desempenho]
    
    # Exibir tabela de métricas
    st.markdown(f"##### Métricas Detalhadas - {tema_sel_desempenho}")
    st.dataframe(
        metricas_filtradas[['Tipo', 'Classe', 'Precision', 'Recall', 'F1-Score', 'Especificidade', 'AUC', 'Acurácia']].style.format({
            'Precision': '{:.3f}',
            'Recall': '{:.3f}',
            'F1-Score': '{:.3f}',
            'Especificidade': '{:.3f}',
            'AUC': '{:.3f}',
            'Acurácia': '{:.3f}'
        }).background_gradient(cmap='RdPu', subset=['Precision', 'Recall', 'F1-Score']),
        use_container_width=True,
        height=300
    )
    
    # Cards de resumo
    st.markdown("##### 📈 Resumo das Métricas")
    col1, col2, col3, col4 = st.columns(4)
    
    acuracia_media = metricas_filtradas['Acurácia'].mean()
    f1_media = metricas_filtradas['F1-Score'].mean()
    precision_media = metricas_filtradas['Precision'].mean()
    recall_media = metricas_filtradas['Recall'].mean()
    
    with col1:
        st.metric("Acurácia Média", f"{acuracia_media:.2%}")
    with col2:
        st.metric("F1-Score Médio", f"{f1_media:.3f}")
    with col3:
        st.metric("Precision Média", f"{precision_media:.3f}")
    with col4:
        st.metric("Recall Médio", f"{recall_media:.3f}")
    
    st.markdown("---")
    
    # ==================== 2. GRÁFICO DE BARRAS - COMPARATIVO ====================
    st.markdown("#### 📊 Comparativo Visual das Métricas")
    
    # Preparar dados para gráfico
    metricas_long = metricas_filtradas.melt(
        id_vars=['Classe', 'Tipo'],
        value_vars=['Precision', 'Recall', 'F1-Score'],
        var_name='Métrica',
        value_name='Score'
    )
    
    # Criar gráfico com ou sem facetas
    if tipo_sel_desempenho == "Todos":
        chart_metricas = alt.Chart(metricas_long).mark_bar().encode(
            x=alt.X('Classe:N', title='Classe'),
            y=alt.Y('Score:Q', title='Score', scale=alt.Scale(domain=[0, 1])),
            color=alt.Color('Métrica:N', scale=alt.Scale(scheme='category10'), legend=alt.Legend(title='Métrica')),
            xOffset='Métrica:N',
            column=alt.Column('Tipo:N', title='Tipo de Texto'),
            tooltip=['Classe', 'Tipo', 'Métrica', alt.Tooltip('Score:Q', format='.3f')]
        ).properties(
            height=350,
            title=f'Comparativo de Métricas - {tema_sel_desempenho}'
        )
    else:
        chart_metricas = alt.Chart(metricas_long).mark_bar().encode(
            x=alt.X('Classe:N', title='Classe'),
            y=alt.Y('Score:Q', title='Score', scale=alt.Scale(domain=[0, 1])),
            color=alt.Color('Métrica:N', scale=alt.Scale(scheme='category10'), legend=alt.Legend(title='Métrica')),
            xOffset='Métrica:N',
            tooltip=['Classe', 'Tipo', 'Métrica', alt.Tooltip('Score:Q', format='.3f')]
        ).properties(
            width=600,
            height=350,
            title=f'Comparativo de Métricas - {tema_sel_desempenho} ({tipo_sel_desempenho})'
        )
    
    st.altair_chart(chart_metricas, use_container_width=True)
    
    # ==================== 3. MATRIZ DE CONFUSÃO ====================
    st.markdown("#### 🔲 Matriz de Confusão")
    
    col1, col2 = st.columns(2)
    
    with col1:
        tema_conf = st.selectbox(
            "Selecione o tema:", 
            list(ARQUIVOS_DATASET.keys()), 
            key="tema_confusao"
        )
    
    with col2:
        tipo_conf = st.selectbox(
            "Selecione o tipo de dado:",
            ["Postagens", "Comentários"],
            key='tipo_confusao'
        )
    
    # Carregar dados
    arquivos = ARQUIVOS_DATASET[tema_conf]
    arquivo_key = "posts_amostra" if tipo_conf == "Postagens" else "comentarios_amostra"
    df_conf = load_data(arquivos[arquivo_key], tipo="amostra")
    
    if not df_conf.empty and {"rotulo", "Classe Sentimento"}.issubset(df_conf.columns):
        
        y_true = df_conf["rotulo"]
        y_pred = df_conf["Classe Sentimento"]
        
        matriz = confusion_matrix(y_true, y_pred, labels=["NEG", "NEU", "POS"])
        
        # Preparar dados para Altair
        matriz_df = (
            pd.DataFrame(matriz, index=["NEG", "NEU", "POS"], columns=["NEG", "NEU", "POS"])
            .reset_index()
            .melt(id_vars='index', var_name='Predito', value_name='Quantidade')
            .rename(columns={'index': 'Verdadeiro'})
        )
        
        # Gráfico de heatmap
        conf_chart = alt.Chart(matriz_df).mark_rect().encode(
            x=alt.X('Predito:N', title='Predito'),
            y=alt.Y('Verdadeiro:N', title='Verdadeiro'),
            color=alt.Color('Quantidade:Q', scale=alt.Scale(scheme='blues'), legend=alt.Legend(title='Quantidade')),
            tooltip=['Verdadeiro', 'Predito', 'Quantidade']
        ).properties(
            width=400,
            height=400,
            title=f'Matriz de Confusão – {tema_conf} ({tipo_conf})'
        )
        
        # Texto sobreposto
        texto = alt.Chart(matriz_df).mark_text(
            fontSize=20,
            color="white",
            fontWeight='bold'
        ).encode(
            x='Predito:N',
            y='Verdadeiro:N',
            text='Quantidade:Q'
        )
        
        st.altair_chart(conf_chart + texto, use_container_width=False)
        
        # Análise da matriz
        col1, col2, col3 = st.columns(3)
        
        diagonal = matriz.diagonal()
        total = matriz.sum()
        acertos = diagonal.sum()
        
        with col1:
            st.metric("Total de Predições", f"{total}")
        with col2:
            st.metric("Acertos", f"{acertos}", delta=f"{acertos/total:.1%}")
        with col3:
            st.metric("Erros", f"{total - acertos}", delta=f"-{(total-acertos)/total:.1%}")
        
    else:
        st.warning("⚠️ Dados insuficientes para exibir a matriz de confusão.")
    
    st.markdown("---")
    
    # ==================== 4. CURVAS ROC ====================
    st.markdown("#### 📈 Curvas ROC (Receiver Operating Characteristic)")
    
    st.markdown("""
    <div class="story-section">
        <div class="story-text">
        As <strong>Curvas ROC</strong> mostram a relação entre a Taxa de Verdadeiros Positivos (TPR) 
        e a Taxa de Falsos Positivos (FPR) em diferentes limiares de classificação. 
        Quanto mais próxima a curva estiver do canto superior esquerdo, melhor o desempenho do modelo.
        <br><br>
        <strong>AUC (Area Under Curve)</strong>: Área sob a curva ROC. Valores próximos a 1.0 indicam excelente desempenho.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        tema_roc = st.selectbox(
            "Tema para análise ROC:", 
            list(ARQUIVOS_DATASET.keys()), 
            key="tema_roc"
        )
    
    with col2:
        tipo_roc = st.selectbox(
            "Tipo de dado:",
            ["Postagens", "Comentários"],
            key='tipo_roc'
        )
    
    # Função para plotar ROC com Altair
    def plot_roc_altair(df, titulo):
        classes = ["NEG", "NEU", "POS"]
        
        # Verificar se existem colunas de probabilidade
        if not all(col in df.columns for col in ["prob_NEG", "prob_NEU", "prob_POS"]):
            st.warning("⚠️ Probabilidades não disponíveis para este conjunto de dados.")
            return
        
        y_true = df["rotulo"]
        y_score = df[["prob_NEG", "prob_NEU", "prob_POS"]]
        y_bin = label_binarize(y_true, classes=classes)
        
        # Construir dataframe para Altair
        roc_list = []
        for i, cls in enumerate(classes):
            try:
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_score.iloc[:, i])
                auc_cls = roc_auc_score(y_bin[:, i], y_score.iloc[:, i])
                for x, y in zip(fpr, tpr):
                    roc_list.append({
                        "FPR": x, 
                        "TPR": y, 
                        "Classe": f"{cls} (AUC = {auc_cls:.3f})"
                    })
            except Exception as e:
                st.warning(f"Erro ao calcular ROC para classe {cls}: {e}")
                continue
        
        if not roc_list:
            st.error("Não foi possível gerar curvas ROC.")
            return
        
        df_roc = pd.DataFrame(roc_list)
        
        # Gráfico principal ROC
        chart = alt.Chart(df_roc).mark_line(strokeWidth=3).encode(
            x=alt.X("FPR:Q", title="Taxa de Falsos Positivos (FPR)", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("TPR:Q", title="Taxa de Verdadeiros Positivos (TPR)", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Classe:N", legend=alt.Legend(title="Classe")),
            tooltip=["Classe", alt.Tooltip("FPR:Q", format=".3f"), alt.Tooltip("TPR:Q", format=".3f")]
        )
        
        # Linha diagonal de referência (classificador aleatório)
        diag = pd.DataFrame({"FPR": [0, 1], "TPR": [0, 1]})
        diag_chart = alt.Chart(diag).mark_line(
            color="gray", 
            strokeDash=[5,5], 
            strokeWidth=2
        ).encode(
            x="FPR",
            y="TPR"
        )
        
        final_chart = (chart + diag_chart).properties(
            title=f"Curvas ROC – {titulo}",
            height=500
        )
        
        st.altair_chart(final_chart, use_container_width=True)
    
    # Carregar dados para ROC
    arquivos_roc = ARQUIVOS_DATASET[tema_roc]
    arquivo_key_roc = "posts_amostra" if tipo_roc == "Postagens" else "comentarios_amostra"
    df_roc = load_data(arquivos_roc[arquivo_key_roc], tipo="amostra")
    
    if not df_roc.empty and "rotulo" in df_roc.columns:
        plot_roc_altair(df_roc, f"{tema_roc} - {tipo_roc}")
        
        # Interpretação da curva ROC
        st.markdown("##### 💡 Interpretação dos Resultados")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("""
            **AUC = 1.0**  
            Classificador perfeito
            """)
        
        with col2:
            st.info("""
            **AUC = 0.7 - 0.9**  
            Bom desempenho
            """)
        
        with col3:
            st.info("""
            **AUC = 0.5**  
            Classificador aleatório (linha diagonal)
            """)
    else:
        st.warning("⚠️ Dados insuficientes para gerar curvas ROC.")
    
    st.markdown("---")
    
    # ==================== 5. ANÁLISE COMPARATIVA ENTRE TEMAS ====================
    st.markdown("#### 🔬 Análise Comparativa entre Temas")
    
    # Gráfico comparativo de F1-Score por classe
    f1_comparison = metricas_completas.copy()
    
    chart_comparison = alt.Chart(f1_comparison).mark_bar().encode(
        x=alt.X('Tema:N', title='Tema'),
        y=alt.Y('F1-Score:Q', title='F1-Score', scale=alt.Scale(domain=[0, 1])),
        color=alt.Color('Classe:N', scale=alt.Scale(domain=['NEG', 'NEU', 'POS'], 
                                                     range=['#ff006e', '#00d4ff', '#00f5a0'])),
        xOffset='Classe:N',
        column=alt.Column('Tipo:N', title='Tipo de Texto'),
        tooltip=['Tema', 'Tipo', 'Classe', alt.Tooltip('F1-Score:Q', format='.3f')]
    ).properties(
        height=350,
        title='Comparação de F1-Score entre Todos os Temas'
    )
    
    st.altair_chart(chart_comparison, use_container_width=True)
    
    # Insights finais
    st.markdown("#### 🎯 Principais Conclusões")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="story-section">
            <div class="story-title">Pontos Fortes do Modelo</div>
            <div class="story-text">
            • <strong>Classe Neutra</strong>: F1-score entre 0.73 e 0.92<br>
            • <strong>Classe Negativa</strong>: Desempenho equilibrado<br>
            • <strong>Postagens</strong>: Acurácia até 84% (texto objetivo)<br>
            • Especificidade alta para classe positiva (até 99%)
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="story-section">
            <div class="story-title">Desafios Identificados</div>
            <div class="story-text">
            • <strong>Classe Positiva</strong>: Recall baixo (20-50%)<br>
            • Desbalanceamento dos dados impacta predições
            </div>
        </div>
        """, unsafe_allow_html=True)
#================================tab 4: EVOLUÇÃO TEMPORAL ====================
with tab4:
    st.markdown("### Evolução Temporal das Opiniões Por Tema")
    
    # ==================== 1. EVOLUÇÃO HISTÓRICA TOTAL - TODOS OS TEMAS ====================
    st.markdown("#### Evolução Histórica das Postagens e Comentários - Todos os Temas")
    
    @st.cache_data
    def gerar_evolucao_unificada():
        """Gera evolução temporal de todos os temas (postagens + comentários) em um único gráfico"""
        evolucao_dados = []
        
        for tema, arquivos in ARQUIVOS_DATASET.items():
            # POSTAGENS
            df_posts = load_data(arquivos['posts'], tipo="completo")
            
            coluna_data = None
            for col in df_posts.columns:
                if any(keyword in col.lower() for keyword in ['date', 'data', 'created', 'timestamp']):
                    coluna_data = col
                    break
            
            if coluna_data and coluna_data in df_posts.columns:
                try:
                    df_posts[coluna_data] = pd.to_datetime(df_posts[coluna_data], errors='coerce')
                    df_posts = df_posts.dropna(subset=[coluna_data])
                    
                    if len(df_posts) > 0:
                        temp = df_posts.groupby(df_posts[coluna_data].dt.to_period("M")).size().reset_index()
                        temp.columns = ["Mes", "Quantidade"]
                        temp["Mes"] = temp["Mes"].astype(str)
                        temp["Tema"] = tema
                        temp["Tipo"] = "Postagens"
                        evolucao_dados.append(temp)
                except Exception as e:
                    st.warning(f"Erro ao processar postagens de {tema}: {e}")
                    continue
            
            # COMENTÁRIOS
            df_comments = load_data(arquivos['comentarios'], tipo="completo")
            
            coluna_data_comm = None
            for col in df_comments.columns:
                if any(keyword in col.lower() for keyword in ['date', 'data', 'created', 'timestamp']):
                    coluna_data_comm = col
                    break
            
            if coluna_data_comm and coluna_data_comm in df_comments.columns:
                try:
                    df_comments[coluna_data_comm] = pd.to_datetime(df_comments[coluna_data_comm], errors='coerce')
                    df_comments = df_comments.dropna(subset=[coluna_data_comm])
                    
                    if len(df_comments) > 0:
                        temp = df_comments.groupby(df_comments[coluna_data_comm].dt.to_period("M")).size().reset_index()
                        temp.columns = ["Mes", "Quantidade"]
                        temp["Mes"] = temp["Mes"].astype(str)
                        temp["Tema"] = tema
                        temp["Tipo"] = "Comentários"
                        evolucao_dados.append(temp)
                except Exception as e:
                    st.warning(f"Erro ao processar comentários de {tema}: {e}")
                    continue
        
        if evolucao_dados:
            return pd.concat(evolucao_dados, ignore_index=True)
        else:
            return pd.DataFrame()
    
    df_evo = gerar_evolucao_unificada()
    
    if not df_evo.empty:
        # ==================== FILTROS INTERATIVOS ====================    
        col_filtro1, col_filtro2 = st.columns([1, 2])
        
        with col_filtro1:
            tipo_visualizacao = st.radio(
                "Selecione o tipo de publicação:",
                options=["📝 Postagens", "💬 Comentários", "📊 Ambos"],
                index=0,
                key="tipo_viz_evolucao",
                horizontal=False
            )
        
        with col_filtro2:
            st.markdown("**Filtrar temas (selecione um ou mais):**")
            
            col_check1, col_check2, col_check3 = st.columns(3)
            
            with col_check1:
                check_stf = st.checkbox("STF", value=True, key="check_stf_evo")
            with col_check2:
                check_auxilio = st.checkbox("Auxílio Brasil", value=True, key="check_auxilio_evo")
            with col_check3:
                check_vacinacao = st.checkbox("Vacinação", value=True, key="check_vacinacao_evo")
        
        # Criar lista de temas selecionados
        temas_selecionados = []
        if check_stf:
            temas_selecionados.append("STF")
        if check_auxilio:
            temas_selecionados.append("Auxílio Brasil")
        if check_vacinacao:
            temas_selecionados.append("Vacinação")
        
        # Verificar se pelo menos um tema foi selecionado
        if not temas_selecionados:
            st.warning("⚠️ Selecione pelo menos um tema para visualizar o gráfico.")
        else:
            # Filtrar dados baseado nas seleções
            df_evo_filtrado = df_evo[df_evo['Tema'].isin(temas_selecionados)].copy()
            
            # Filtrar por tipo de publicação
            if tipo_visualizacao == "📝 Postagens":
                df_evo_filtrado = df_evo_filtrado[df_evo_filtrado['Tipo'] == 'Postagens']
                titulo_grafico = 'Evolução Mensal das Postagens'
                mostrar_legenda_tipo = False
            elif tipo_visualizacao == "💬 Comentários":
                df_evo_filtrado = df_evo_filtrado[df_evo_filtrado['Tipo'] == 'Comentários']
                titulo_grafico = 'Evolução Mensal dos Comentários'
                mostrar_legenda_tipo = False
            else:  # Ambos
                titulo_grafico = 'Evolução Mensal das Postagens e Comentários'
                mostrar_legenda_tipo = True
            
            # Criar identificador único para legenda
            if mostrar_legenda_tipo:
                df_evo_filtrado['Legenda'] = df_evo_filtrado['Tema'] + ' (' + df_evo_filtrado['Tipo'] + ')'
            else:
                df_evo_filtrado['Legenda'] = df_evo_filtrado['Tema']
            
            # ==================== GRÁFICO FILTRADO ====================
            chart = alt.Chart(df_evo_filtrado).mark_line(point=True, strokeWidth=3).encode(
                x=alt.X('Mes:N', 
                       title='Mês', 
                       axis=alt.Axis(labelAngle=-45, labelFontSize=10)),
                y=alt.Y('Quantidade:Q', 
                       title='Número de Publicações',
                       axis=alt.Axis(labelFontSize=11)),
                color=alt.Color('Legenda:N', 
                              scale=alt.Scale(scheme='category10'),
                              legend=alt.Legend(
                                  title='Tema' if not mostrar_legenda_tipo else 'Tema e Tipo',
                                  titleFontSize=12,
                                  labelFontSize=10,
                                  symbolSize=150
                              )),
                tooltip=['Mes:N', 'Tema:N', 'Tipo:N', alt.Tooltip('Quantidade:Q', format=',')]
            )
            
            # Adicionar strokeDash apenas se estiver mostrando ambos
            if mostrar_legenda_tipo:
                chart = chart.encode(
                    strokeDash=alt.StrokeDash('Tipo:N',
                                             scale=alt.Scale(
                                                 domain=['Postagens', 'Comentários'],
                                                 range=[[1], [5, 3]]
                                             ),
                                             legend=alt.Legend(
                                                 title='Tipo de Publicação',
                                                 titleFontSize=12,
                                                 labelFontSize=10,
                                                 symbolSize=150,
                                                 symbolStrokeWidth=3
                                             ))
                )
            
            chart = chart.properties(
                height=500,
                title={
                    "text": titulo_grafico,
                    "fontSize": 15,
                    "fontWeight": "bold"
                }
            ).interactive()
            
            st.altair_chart(chart, use_container_width=True)
            
            # ==================== ESTATÍSTICAS DINÂMICAS ====================
            st.markdown("---")
            
            # Estatísticas de POSTAGENS (se selecionadas)
            if tipo_visualizacao in ["📝 Postagens", "📊 Ambos"]:
                st.markdown("#### 📊 Estatísticas por Tema - Postagens")
                
                cols_posts = st.columns(len(temas_selecionados))
                
                for idx, tema in enumerate(temas_selecionados):
                    dados_tema = df_evo[(df_evo['Tema'] == tema) & (df_evo['Tipo'] == 'Postagens')]
                    
                    if not dados_tema.empty:
                        total = dados_tema['Quantidade'].sum()
                        media = dados_tema['Quantidade'].mean()
                        pico = dados_tema['Quantidade'].max()
                        mes_pico = dados_tema[dados_tema['Quantidade'] == pico]['Mes'].values[0] if len(dados_tema[dados_tema['Quantidade'] == pico]) > 0 else 'N/A'
                        
                        with cols_posts[idx]:
                            st.markdown(f"""
                            <div class="metric-card" style="background: linear-gradient(135deg, #1e3a5f 0%, #2d1b4e 100%);">
                                <div class="metric-label">{tema}</div>
                                <div class="metric-value">{total:,}</div>
                                <div class="metric-label">Posts no período</div>
                                <hr style="border-color: rgba(255,255,255,0.1); margin: 0.5rem 0;">
                                <small style="color: #a0a0a0;">
                                📊 Média: {media:.1f}/mês<br>
                                🔝 Pico: {pico} em {mes_pico}
                                </small>
                            </div>
                            """, unsafe_allow_html=True)
            
            # Estatísticas de COMENTÁRIOS (se selecionadas)
            if tipo_visualizacao in ["💬 Comentários", "📊 Ambos"]:
                st.markdown("#### 💬 Estatísticas por Tema - Comentários")
                
                cols_comments = st.columns(len(temas_selecionados))
                
                for idx, tema in enumerate(temas_selecionados):
                    dados_tema = df_evo[(df_evo['Tema'] == tema) & (df_evo['Tipo'] == 'Comentários')]
                    
                    if not dados_tema.empty:
                        total = dados_tema['Quantidade'].sum()
                        media = dados_tema['Quantidade'].mean()
                        pico = dados_tema['Quantidade'].max()
                        mes_pico = dados_tema[dados_tema['Quantidade'] == pico]['Mes'].values[0] if len(dados_tema[dados_tema['Quantidade'] == pico]) > 0 else 'N/A'
                        
                        with cols_comments[idx]:
                            st.markdown(f"""
                            <div class="metric-card" style="background: linear-gradient(135deg, #4a2c2a 0%, #3d2a1f 100%); border: 1px solid rgba(255, 140, 0, 0.3);">
                                <div class="metric-label" style="color: #ffb366;">{tema}</div>
                                <div class="metric-value" style="color: #ff8c42;">{total:,}</div>
                                <div class="metric-label" style="color: #cc9966;">Comentários no período</div>
                                <hr style="border-color: rgba(255,140,0,0.2); margin: 0.5rem 0;">
                                <small style="color: #cc9966;">
                                📊 Média: {media:.1f}/mês<br>
                                🔝 Pico: {pico} em {mes_pico}
                                </small>
                            </div>
                            """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Não foi possível gerar evolução temporal. Verifique as colunas de data nos arquivos.")
    
    st.markdown("---")
    
    # ==================== 2. EVOLUÇÃO DOS SENTIMENTOS - TODOS OS TEMAS UNIFICADOS ====================
    st.markdown("#### 📅 Evolução Temporal dos Sentimentos - Análise Individual")
    
    col1, col2 = st.columns(2)
    
    with col1:
        tema_sel = st.selectbox("Selecione o tema:", list(ARQUIVOS_DATASET.keys()), key="tema_evolucao")
    
    with col2:
        tipo_sel = st.selectbox(
            "Selecione o tipo de dado:",
            ["Postagens", "Comentários"],
            key='tipo_evolucao'
        )
    
    # Carregar dados do tema selecionado
    arquivos = ARQUIVOS_DATASET[tema_sel]
    arquivo_key = "posts" if tipo_sel == "Postagens" else "comentarios"
    df = load_data(arquivos[arquivo_key], tipo="completo")
    
    # Procurar coluna de data
    coluna_data = None
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['date', 'data', 'created', 'timestamp']):
            coluna_data = col
            break
    
    if coluna_data and coluna_data in df.columns and 'Classe Sentimento' in df.columns:
        try:
            df[coluna_data] = pd.to_datetime(df[coluna_data], errors='coerce')
            df = df.dropna(subset=[coluna_data])
            
            if len(df) > 0:
                df['Ano-Mês'] = df[coluna_data].dt.to_period('M').dt.to_timestamp()
                
                evolucao_sent = (
                    df.groupby(['Ano-Mês', 'Classe Sentimento'])
                      .size()
                      .reset_index(name='Quantidade')
                )
                
                # Gráfico de linhas por sentimento
                linha_sent = alt.Chart(evolucao_sent).mark_line(point=True, strokeWidth=2).encode(
                    x=alt.X('Ano-Mês:T', title='Data', axis=alt.Axis(format='%b %Y')),
                    y=alt.Y('Quantidade:Q', title='Quantidade'),
                    color=alt.Color('Classe Sentimento:N',
                                  scale=alt.Scale(domain=['NEG', 'NEU', 'POS'],
                                                range=['#ff006e', '#00d4ff', '#00f5a0']),
                                  legend=alt.Legend(title='Sentimento')),
                    tooltip=[
                        alt.Tooltip('Ano-Mês:T', format='%B %Y', title='Mês'),
                        alt.Tooltip('Classe Sentimento:N', title='Sentimento'),
                        alt.Tooltip('Quantidade:Q', title='Quantidade')
                    ]
                ).properties(
                    height=450,
                    title=f'Evolução dos Sentimentos - {tema_sel} ({tipo_sel})'
                )
                
                st.altair_chart(linha_sent, use_container_width=True)
                
                # Análise de tendências
                st.markdown("#### 💡 Análise de Tendências")
                
                # Calcular percentuais atuais vs anteriores
                periodo_recente = df[df[coluna_data] >= df[coluna_data].max() - pd.DateOffset(months=6)]
                periodo_anterior = df[(df[coluna_data] < df[coluna_data].max() - pd.DateOffset(months=6)) & 
                                     (df[coluna_data] >= df[coluna_data].max() - pd.DateOffset(months=12))]
                
                col1, col2, col3 = st.columns(3)
                
                for idx, sent in enumerate(['NEG', 'NEU', 'POS']):
                    recente = len(periodo_recente[periodo_recente['Classe Sentimento'] == sent])
                    anterior = len(periodo_anterior[periodo_anterior['Classe Sentimento'] == sent])
                    
                    variacao = ((recente - anterior) / anterior * 100) if anterior > 0 else 0
                    
                    sentimento_label = {'NEG': '🔴 Negativo', 'NEU': '⚪ Neutro', 'POS': '🟢 Positivo'}[sent]
                    
                    with [col1, col2, col3][idx]:
                        st.metric(
                            label=sentimento_label,
                            value=f"{recente}",
                            delta=f"{variacao:+.1f}% vs 6 meses atrás"
                        )
            else:
                st.warning("Não há dados suficientes para análise temporal.")
        except Exception as e:
            st.error(f"Erro ao processar dados temporais: {e}")
    else:
        st.warning(f"⚠️ Coluna de data ou 'Classe Sentimento' não encontrada para {tema_sel} ({tipo_sel}).")
        st.info("💡 Colunas esperadas: 'date', 'data', 'Data', 'created_at' ou similar")
    

# ==================== RODAPÉ ====================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #a0a0a0; padding: 2rem;">
    <strong>O Retrato Digital da Opinião Pública Brasileira</strong><br>
    Análise de Sentimentos no Reddit | 2015-2025<br>
    Modelo: BERTweet.br via Pysentimento<br>
    <br>
    <em>"Entender a opinião pública é decifrar um país em movimento."</em>
</div>
<div style='text-align: center; color: #FFFFFF; font-size: 0.9rem; background-color: #000000; padding: 1rem; border-radius: 8px;'> 
    <footer> Desenvolvido Por Sávio Sousa • © 2025 </footer>
</div>
""", unsafe_allow_html=True)