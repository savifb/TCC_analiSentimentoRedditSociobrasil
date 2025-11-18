
import plotly.express as px
import pandas as pd

def pie_sentiment(df):
    s = df['sentiment'].value_counts().reset_index()
    s.columns = ['sentiment','count']
    fig = px.pie(s, names='sentiment', values='count', title='Distribuição de Sentimentos')
    return fig

def bar_sentiment(df, groupby='sentiment'):
    s = df.groupby([groupby]).size().reset_index(name='count')
    # If grouped by sentiment use colors by sentiment (simple)
    if groupby == 'sentiment':
        fig = px.bar(s, x=groupby, y='count', title='Contagem por sentimento')
    else:
        # assume df has 'sentiment' column, produce grouped bar
        s2 = df.groupby([groupby, 'sentiment']).size().reset_index(name='count')
        fig = px.bar(s2, x=groupby, y='count', color='sentiment', barmode='group', title=f'Contagem por {groupby} e sentimento')
    return fig

def time_series_sentiment(df, date_col='date'):
    if date_col not in df.columns:
        return {}
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    s = df.groupby([pd.Grouper(key=date_col, freq='W'), 'sentiment']).size().reset_index(name='count')
    fig = px.line(s, x=date_col, y='count', color='sentiment', title='Série temporal por sentimento (semanal)')
    return fig

def compare_posts_comments(df):
    s = df.groupby(['type','sentiment']).size().reset_index(name='count')
    fig = px.bar(s, x='type', y='count', color='sentiment', barmode='group', title='Posts vs Comentários por sentimento')
    return fig

def top_n_by_sentiment(df, sentiment='negative', n=10):
    # simple example: count words in texts for the given sentiment
    df2 = df[df['sentiment']==sentiment].copy()
    if df2.empty:
        return []
    texts = df2['text'].dropna().str.lower().str.replace('[^a-z0-9\s]', ' ', regex=True)
    words = texts.str.split().explode().value_counts().head(n).reset_index()
    words.columns = ['term','count']
    return words

def top_n_by_count(df, n=10):
    # try to use 'topic' column if present, otherwise top words
    if 'topic' in df.columns:
        return df['topic'].value_counts().head(n).reset_index().rename(columns={'index':'topic','topic':'count'})
    texts = df['text'].dropna().str.lower().str.replace('[^a-z0-9\s]', ' ', regex=True)
    words = texts.str.split().explode().value_counts().head(n).reset_index()
    words.columns = ['term','count']
    return words
