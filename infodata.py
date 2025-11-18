import pandas as pd 
DATA_PATH = "data/"

amostracompletaSTFPosts = pd.read_csv(DATA_PATH + "amostracompletaSTFPosts.csv", sep=";")
amostracompletaSTFComentarios = pd.read_csv(DATA_PATH + "amostraCompletaSTFComentarios.csv", sep=";")

amostraCompletaABPosts = pd.read_csv(DATA_PATH + "amostraCompletaABPosts.csv", sep=";")
amostraCompletaABComentarios = pd.read_csv(DATA_PATH + "amostraCompletaABComentarios.csv", sep=";")

amostraCompletoVSPosts1 = pd.read_csv(DATA_PATH + "amostraCompletoVSPosts1.csv", sep=";")
amostraCompletoVSComentarios1 = pd.read_csv(DATA_PATH + "amostraCompletoVSComentarios1.csv", sep=";")

datasets = {
    "amostra_completaSTFPosts": amostracompletaSTFPosts,
    "amostra_completaSTFComentarios": amostracompletaSTFComentarios,
    "amostraCompletaABPosts": amostraCompletaABPosts,
    "amostraCompletaABComentarios": amostraCompletaABComentarios,
    "amostraVSCompletoPosts": amostraCompletoVSPosts1,
    "amostraVSCompletoComentarios": amostraCompletoVSComentarios1
}

# Padronização
for name, df in datasets.items():
    for col in ['Classe Sentimento', 'rotulo']:
        if col in df.columns:

            df[col] = (
                df[col]
                .fillna('NEU')
                .astype(str)
                .replace({
                    'neu': 'NEU',
                    'NEY': 'NEU',
                    'UNKNOWN': 'NEU',
                    'MEI': 'NEU',
                    'NaN': 'NEU',
                    'BEG': 'NEG',
                    'BEY': 'NEU'
                })
            )

    print(f"Standardization complete for {name}.")

# Contagem dos rótulos — substituir o que estava errado
for name, df in datasets.items():
    print(f"\n{name} – contagem dos rótulos:")
    if "rotulo" in df.columns:
        print(df["rotulo"].value_counts())


# Colunas de cada amostra 

for nome, df in datasets.items():
    print(f"\nColunas em {nome}:")
    print(df.columns.tolist())