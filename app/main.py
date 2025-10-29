import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text) # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text) # Remove punctuation
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# 1. Inicializa o app FastAPI
app = FastAPI(title="API de Análise de Sentimento", version="1.0")

# 2. Carrega o pipeline do modelo (só uma vez, quando a app inicia)
try:                                
    pipeline = joblib.load("models/sentiment_pipeline.pkl")
    print("Modelo carregado com sucesso!")
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")
    pipeline = None

# 3. Define o "formato" de entrada dos dados (validação)
class ReviewInput(BaseModel):
    text: str

# 4. Define o "formato" de saída dos dados
class PredictionOutput(BaseModel):
    sentiment: str
    label: int # 0 ou 1

# 5. Define o endpoint de "health check"
@app.get("/", tags=["Health Check"])
def read_root():
    return {"status": "API está no ar!"}

# 6. Define o endpoint de predição
@app.post("/predict", tags=["Predição"], response_model=PredictionOutput)
def predict_sentiment(review: ReviewInput):
    if pipeline is None:
        return {"error": "Modelo não foi carregado."}

    # Limpa o texto de entrada (IMPORTANTE: usar a *mesma* limpeza do treino)
    cleaned_text = clean_text(review.text)

    # Faz a predição (o pipeline aplica o TF-IDF e o modelo)
    # model.predict() espera uma lista, por isso [cleaned_text]
    prediction = pipeline.predict([cleaned_text])
    label = int(prediction[0])
    sentiment = "positive" if label == 1 else "negative"

    return {"sentiment": sentiment, "label": label}

# 7. (Opcional) Permite rodar o script diretamente
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)