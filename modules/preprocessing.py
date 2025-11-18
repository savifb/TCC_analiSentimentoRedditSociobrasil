
import pandas as pd
import numpy as np

def load_sample(path='data/sample_dataset.csv'):
    df = pd.read_csv(path)
    return df

def basic_clean(df):
    df = df.copy()
    # remove colunas pessoais se existirem
    for c in ['user','username','id']:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)
    df.dropna(subset=['text'], inplace=True)
    return df
