
import joblib, os

MODEL_PATH = os.path.join('models','sentiment_model.joblib')

def load_model(path=MODEL_PATH):
    if os.path.exists(path):
        return joblib.load(path)
    return None

def predict_texts(texts, model=None):
    m = model or load_model()
    if m is None:
        raise RuntimeError('Modelo não encontrado. Coloque em models/sentiment_model.joblib')
    return m.predict(texts)
