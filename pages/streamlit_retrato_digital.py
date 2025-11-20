import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime

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
    
    /* Navegação de seções */
    .nav-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.8rem 2rem;
        border-radius: 25px;
        border: none;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        margin: 0.5rem;
    }
    
    .nav-button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
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
    
    /* Tabelas estilizadas */
    .dataframe {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
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

# ==================== FUNÇÕES DE CARREGAMENTO DE DADOS ====================
@st.cache_data
def create_sample_data():
    """Cria dados de exemplo para demonstração"""
    
    # Dados de polaridade - Postagens
    posts_data = {
        'Tema': ['STF', 'Auxílio Brasil', 'Vacinação Covid'],
        'Negativo': [76, 57, 76],
        'Neutro': [400, 371, 665],
        'Positivo': [30, 9, 8]
    }
    
    # Dados de polaridade - Comentários
    comments_data = {
        'Tema': ['STF', 'Auxílio Brasil', 'Vacinação Covid'],
        'Negativo': [300, 286, 441],
        'Neutro': [200, 248, 680],
        'Positivo': [50, 28, 67]
    }
    
    # Métricas de desempenho
    metrics_data = {
        'Tema': ['STF', 'STF', 'STF', 'Auxílio', 'Auxílio', 'Auxílio', 'Vacina', 'Vacina', 'Vacina'],
        'Tipo': ['Comentários']*3 + ['Postagens']*3 + ['Postagens']*3,
        'Classe': ['Negativo', 'Neutro', 'Positivo']*3,
        'Precision': [0.78, 0.82, 0.65, 0.54, 0.92, 0.20, 0.84, 0.88, 0.16],
        'Recall': [0.76, 0.85, 0.59, 0.60, 0.90, 0.30, 0.82, 0.86, 0.20],
        'F1-Score': [0.78, 0.82, 0.65, 0.54, 0.92, 0.20, 0.84, 0.88, 0.16],
        'Acurácia': [0.7912]*3 + [0.8534]*3 + [0.8314]*3
    }
    
    # Série temporal
    dates = pd.date_range('2019-01', '2025-06', freq='M')
    ts_data = []
    for tema in ['STF', 'Auxílio Brasil', 'Vacinação Covid']:
        for date in dates:
            for pol in ['Negativo', 'Neutro', 'Positivo']:
                prob = {'Negativo': 0.35, 'Neutro': 0.55, 'Positivo': 0.10}[pol]
                count = np.random.poisson(prob * 100)
                ts_data.append({'Data': date, 'Tema': tema, 'Polaridade': pol, 'Contagem': count})
    
    # Matriz de confusão
    cm_data = {
        'Verdadeiro': ['Negativo']*3 + ['Neutro']*3 + ['Positivo']*3,
        'Predito': ['Negativo', 'Neutro', 'Positivo']*3,
        'Quantidade': [120, 30, 10, 40, 300, 60, 5, 20, 90]
    }
    
    return (pd.DataFrame(posts_data), 
            pd.DataFrame(comments_data), 
            pd.DataFrame(metrics_data),
            pd.DataFrame(ts_data),
            pd.DataFrame(cm_data))

# ==================== CARREGAR DADOS ====================
posts_df, comments_df, metrics_df, ts_df, cm_df = create_sample_data()

# ==================== PÁGINA PRINCIPAL ====================
st.markdown('<h1 class="big-title">O Retrato Digital da Opinião Pública Brasileira</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Uma jornada pelos debates que dividem o país • 2015-2025</p>', unsafe_allow_html=True)

# ==================== MÉTRICAS PRINCIPAIS ====================
st.markdown("### 📊 Visão Geral da Pesquisa")
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
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📖 Storytelling", 
    "📈 Polaridades", 
    "🎯 Desempenho do Modelo",
    "📅 Evolução Temporal",
    "🔬 Análise Detalhada"
])

# ==================== TAB 1: STORYTELLING ====================
with tab1:
    st.markdown('<div class="story-section">', unsafe_allow_html=True)
    st.markdown('<div class="story-title">Capítulo 1: O Ponto de Partida</div>', unsafe_allow_html=True)
    st.markdown('''
    <div class="story-text">
    Imagine entrar em uma sala repleta de brasileiros debatendo os temas mais polêmicos do país. 
    Você ouviria vozes exaltadas falando sobre o <span class="highlight-number">Supremo Tribunal Federal</span>, 
    discussões acaloradas sobre <span class="highlight-number">programas sociais</span>, e opiniões divergentes 
    sobre a <span class="highlight-number">vacinação contra a Covid-19</span>. 
    <br><br>
    Essa sala existe — e se chama <strong>Reddit</strong>.
    </div>
    ''', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="story-section">', unsafe_allow_html=True)
        st.markdown('<div class="story-title">O STF: A Instituição que Não Convence</div>', unsafe_allow_html=True)
        st.markdown('''
        <div class="story-text">
        De <strong>fevereiro/2015 a junho/2025</strong> — uma década inteira de conversas sobre o STF.
        <br><br>
        • Postagens neutras <strong>desencadeiam comentários negativos</strong><br>
        • O STF enfrenta <strong>crise de legitimidade digital</strong><br>
        • Negatividade supera amplamente o apoio
        </div>
        ''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="story-section">', unsafe_allow_html=True)
        st.markdown('<div class="story-title">Auxílio Brasil: A Promessa Inacabada</div>', unsafe_allow_html=True)
        st.markdown('''
        <div class="story-text">
        <span class="highlight-number">286 negativas</span> vs <span class="highlight-number">28 positivas</span>
        <br><br>
        • Frustrações com <strong>burocracia e cadastro</strong><br>
        • Questionamentos sobre <strong>elegibilidade</strong><br>
        • Valor considerado <strong>insuficiente</strong><br>
        • Persistência histórica de <strong>58% de negatividade</strong>
        </div>
        ''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="story-section">', unsafe_allow_html=True)
    st.markdown('<div class="story-title">Covid-19: A Vacina Aprovada, o Governo Reprovado</div>', unsafe_allow_html=True)
    st.markdown('''
    <div class="story-text">
    Um paradoxo fascinante: <strong>apoiam a ciência, desconfiam do governo</strong>.
    <br><br>
    • <span class="highlight-number">665 postagens neutras</span> com dados e estatísticas<br>
    • <span class="highlight-number">441 comentários negativos</span> sobre gestão governamental<br>
    • Distinção clara: não é a vacina em julgamento, é a <strong>gestão governamental</strong><br>
    • 67 comentários positivos celebram o <strong>avanço científico</strong>
    </div>
    ''', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ==================== TAB 2: POLARIDADES ====================
with tab2:
    st.markdown("### 📊 Distribuição de Sentimentos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Postagens")
        posts_long = posts_df.melt(id_vars='Tema', var_name='Polaridade', value_name='Quantidade')
        
        chart_posts = alt.Chart(posts_long).mark_bar().encode(
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
        st.markdown("#### Comentários")
        comments_long = comments_df.melt(id_vars='Tema', var_name='Polaridade', value_name='Quantidade')
        
        chart_comments = alt.Chart(comments_long).mark_bar().encode(
            x=alt.X('Tema:N', title='Tema'),
            y=alt.Y('Quantidade:Q', title='Quantidade'),
            color=alt.Color('Polaridade:N',
                          scale=alt.Scale(domain=['Negativo', 'Neutro', 'Positivo'],
                                        range=['#ff006e', '#00d4ff', '#00f5a0'])),
            xOffset='Polaridade:N',
            tooltip=['Tema', 'Polaridade', 'Quantidade']
        ).properties(height=400)
        
        st.altair_chart(chart_comments, use_container_width=True)
    
    # Proporções
    st.markdown("### 🎯 Análise Proporcional")
    tema_selecionado = st.selectbox("Selecione o tema para análise detalhada:", 
                                     ['STF', 'Auxílio Brasil', 'Vacinação Covid'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        idx = ['STF', 'Auxílio Brasil', 'Vacinação Covid'].index(tema_selecionado)
        post_data = pd.DataFrame({
            'Polaridade': ['Negativo', 'Neutro', 'Positivo'],
            'Quantidade': [posts_df.iloc[idx]['Negativo'], 
                          posts_df.iloc[idx]['Neutro'], 
                          posts_df.iloc[idx]['Positivo']]
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
            'Quantidade': [comments_df.iloc[idx]['Negativo'], 
                          comments_df.iloc[idx]['Neutro'], 
                          comments_df.iloc[idx]['Positivo']]
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
    st.markdown("### 🎯 Métricas de Desempenho do Modelo")
    
    # Filtros
    col1, col2 = st.columns(2)
    with col1:
        tema_filter = st.selectbox("Tema:", ['Todos'] + list(metrics_df['Tema'].unique()))
    with col2:
        tipo_filter = st.selectbox("Tipo de Texto:", ['Todos'] + list(metrics_df['Tipo'].unique()))
    
    # Aplicar filtros
    filtered_metrics = metrics_df.copy()
    if tema_filter != 'Todos':
        filtered_metrics = filtered_metrics[filtered_metrics['Tema'] == tema_filter]
    if tipo_filter != 'Todos':
        filtered_metrics = filtered_metrics[filtered_metrics['Tipo'] == tipo_filter]
    
    # Criar label combinada
    filtered_metrics['Label'] = filtered_metrics['Tema'] + ' - ' + filtered_metrics['Classe']
    
    # Gráfico de barras agrupadas usando Altair
    metrics_long = filtered_metrics.melt(
        id_vars=['Label', 'Tema', 'Classe'],
        value_vars=['Precision', 'Recall', 'F1-Score'],
        var_name='Métrica',
        value_name='Score'
    )
    
    chart_metrics = alt.Chart(metrics_long).mark_bar().encode(
        x=alt.X('Label:N', title='Tema e Classe', axis=alt.Axis(labelAngle=-45)),
        y=alt.Y('Score:Q', title='Score', scale=alt.Scale(domain=[0, 1])),
        color=alt.Color('Métrica:N', scale=alt.Scale(scheme='category10')),
        xOffset='Métrica:N',
        tooltip=['Label', 'Métrica', alt.Tooltip('Score:Q', format='.2f')]
    ).properties(height=500, title='Métricas por Tema e Classe')
    
    # Adicionar valores nas barras
    text = alt.Chart(metrics_long).mark_text(
        align='center',
        baseline='bottom',
        dy=-5,
        fontSize=10
    ).encode(
        x=alt.X('Label:N'),
        y=alt.Y('Score:Q'),
        text=alt.Text('Score:Q', format='.2f'),
        xOffset='Métrica:N'
    )
    
    st.altair_chart(chart_metrics + text, use_container_width=True)
    
    # Tabela detalhada
    st.markdown("#### 📋 Tabela Detalhada de Métricas")
    st.dataframe(
        filtered_metrics[['Tema', 'Tipo', 'Classe', 'Precision', 'Recall', 'F1-Score', 'Acurácia']].style.format({
            'Precision': '{:.2f}',
            'Recall': '{:.2f}',
            'F1-Score': '{:.2f}',
            'Acurácia': '{:.2f}'
        }).background_gradient(cmap='viridis', subset=['Precision', 'Recall', 'F1-Score']),
        use_container_width=True,
        height=300
    )
    
    # Matriz de confusão
    st.markdown("### 🔲 Matriz de Confusão (Exemplo: Comentários STF)")
    
    cm_pivot = cm_df.pivot(index='Verdadeiro', columns='Predito', values='Quantidade')
    cm_data = []
    for i, row_name in enumerate(cm_pivot.index):
        for j, col_name in enumerate(cm_pivot.columns):
            cm_data.append({
                'Verdadeiro': row_name,
                'Predito': col_name,
                'Quantidade': cm_pivot.iloc[i, j]
            })
    cm_chart_df = pd.DataFrame(cm_data)
    
    heatmap = alt.Chart(cm_chart_df).mark_rect().encode(
        x=alt.X('Predito:N', title='Predito'),
        y=alt.Y('Verdadeiro:N', title='Verdadeiro'),
        color=alt.Color('Quantidade:Q', scale=alt.Scale(scheme='blues'), legend=alt.Legend(title='Quantidade')),
        tooltip=['Verdadeiro', 'Predito', 'Quantidade']
    ).properties(height=400, width=400)
    
    text_heatmap = alt.Chart(cm_chart_df).mark_text(fontSize=16, color='black').encode(
        x='Predito:N',
        y='Verdadeiro:N',
        text='Quantidade:Q'
    )
    
    st.altair_chart(heatmap + text_heatmap, use_container_width=False)

# ==================== TAB 4: EVOLUÇÃO TEMPORAL ====================
with tab4:
    st.markdown("### 📅 Evolução Temporal das Polaridades")
    
    tema_ts = st.selectbox("Selecione o tema:", ts_df['Tema'].unique(), key='ts_tema')
    
    ts_filtered = ts_df[ts_df['Tema'] == tema_ts]
    
    # Gráfico de área empilhada
    area_chart = alt.Chart(ts_filtered).mark_area(opacity=0.7).encode(
        x=alt.X('Data:T', title='Data'),
        y=alt.Y('Contagem:Q', title='Contagem de Menções', stack='zero'),
        color=alt.Color('Polaridade:N',
                       scale=alt.Scale(domain=['Negativo', 'Neutro', 'Positivo'],
                                     range=['#ff006e', '#00d4ff', '#00f5a0']),
                       legend=alt.Legend(title='Polaridade')),
        tooltip=[
            alt.Tooltip('Data:T', format='%b %Y'),
            'Polaridade:N',
            'Contagem:Q'
        ]
    ).properties(
        height=500,
        title=f'Evolução Temporal - {tema_ts}'
    )
    
    # Adicionar linha sobre a área
    line_chart = alt.Chart(ts_filtered).mark_line(size=2).encode(
        x='Data:T',
        y='Contagem:Q',
        color=alt.Color('Polaridade:N',
                       scale=alt.Scale(domain=['Negativo', 'Neutro', 'Positivo'],
                                     range=['#ff006e', '#00d4ff', '#00f5a0']))
    )
    
    st.altair_chart(area_chart, use_container_width=True)
    
    # Insights
    st.markdown("#### 💡 Insights Temporais")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**STF**: Presença contínua desde 2015, com picos em momentos de decisões controversas")
    with col2:
        st.info("**Auxílio Brasil**: Intensificação dos debates a partir de 2019, com reformulação do programa")
    with col3:
        st.info("**Vacinação**: Explosão de discussões durante pandemia (2020-2022), normalização posterior")

# ==================== TAB 5: ANÁLISE DETALHADA ====================
with tab5:
    st.markdown("### 🔬 Análise Profunda: O Comportamento do Modelo")
    
    st.markdown("""
    <div class="story-section">
        <div class="story-title">O Desafio da Classe Positiva</div>
        <div class="story-text">
        O modelo BERTweet.br apresentou <strong>alta especificidade</strong> (até 99%) para a classe positiva, 
        mas <strong>baixo recall</strong> (20-50%). Isso caracteriza um fenômeno de <span class="highlight-number">subdetecção</span>:
        <br><br>
        • O modelo só classifica como "positivo" quando está <strong>altamente confiante</strong><br>
        • Deixa de reconhecer grande parte das manifestações positivas reais<br>
        • Reflexo do <strong>desbalanceamento</strong> dos dados<br>
        • Elogios são raros, sutis e frequentemente <strong>irônicos</strong> em debates sociopolíticos
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="story-section">
            <div class="story-title">O Viés do Neutro</div>
            <div class="story-text">
            • Classe neutra: <strong>F1-score entre 0.73 e 0.92</strong><br>
            • Modelo tende a alocar textos ambíguos para neutro<br>
            • Reflete natureza informativa das postagens<br>
            • <strong>Zona de conforto algorítmica</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="story-section">
            <div class="story-title">Postagens vs Comentários</div>
            <div class="story-text">
            • <strong>Postagens</strong>: Acurácia até 84% (objetivas)<br>
            • <strong>Comentários</strong>: Acurácia mínima 69% (emotivos)<br>
            • Comentários têm ironia, informalidade<br>
            • Mas captam melhor a <strong>opinião pública real</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Gráfico de barras comparativo por classe
    st.markdown("#### 📊 Comparação de Métricas por Classe")
    
    agg_metrics = metrics_df.groupby('Classe')[['Precision', 'Recall', 'F1-Score']].mean().reset_index()
    agg_long = agg_metrics.melt(id_vars='Classe', var_name='Métrica', value_name='Score')
    
    comparison_chart = alt.Chart(agg_long).mark_bar().encode(
        x=alt.X('Classe:N', title='Classe'),
        y=alt.Y('Score:Q', title='Score Médio', scale=alt.Scale(domain=[0, 1])),
        color=alt.Color('Métrica:N', scale=alt.Scale(scheme='category10')),
        xOffset='Métrica:N',
        tooltip=['Classe', 'Métrica', alt.Tooltip('Score:Q', format='.3f')]
    ).properties(height=400, title='Comparação Multidimensional das Métricas por Classe')
    
    st.altair_chart(comparison_chart, use_container_width=True)
    
    # Análise de acurácia por tema
    st.markdown("#### 🎯 Acurácia por Tema e Tipo de Texto")
    
    accuracy_df = metrics_df[['Tema', 'Tipo', 'Acurácia']].drop_duplicates()
    
    accuracy_chart = alt.Chart(accuracy_df).mark_bar().encode(
        x=alt.X('Tema:N', title='Tema'),
        y=alt.Y('Acurácia:Q', title='Acurácia', scale=alt.Scale(domain=[0.6, 0.9])),
        color=alt.Color('Tipo:N', scale=alt.Scale(scheme='set2')),
        xOffset='Tipo:N',
        tooltip=['Tema', 'Tipo', alt.Tooltip('Acurácia:Q', format='.2%')]
    ).properties(height=350)
    
    st.altair_chart(accuracy_chart, use_container_width=True)

# ==================== RODAPÉ ====================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #a0a0a0; padding: 2rem;">
    <strong>O Retrato Digital da Opinião Pública Brasileira</strong><br>
    Análise de Sentimentos no Reddit | 2015-2025<br>
    Modelo: BERTweet.br via PysentimentoBR<br>
    <br>
    <em>"Os números não mentem, mas também não contam toda a história."</em>
</div>
""", unsafe_allow_html=True)