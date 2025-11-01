import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import re
import nltk
from nltk.corpus import stopwords
import os

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text) 
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

app = FastAPI(title="API de Análise de Sentimento", version="1.0")


model_path_docker = "/models/sentiment_pipeline.pkl"  
model_path_local = "models/sentiment_pipeline.pkl"   

if os.path.exists(model_path_docker):
    model_path = model_path_docker
else:
    model_path = model_path_local

try:
    pipeline = joblib.load(model_path)
    print(f"Modelo carregado com sucesso de: {model_path}")
except Exception as e:
    print(f"Erro ao carregar o modelo de '{model_path}': {e}")
    pipeline = None

class ReviewInput(BaseModel):
    text: str

class PredictionOutput(BaseModel):
    sentiment: str
    label: int 

@app.get("/", tags=["Health Check"])
def read_root():
    return {"status": "API está no ar!"}

@app.post("/predict", tags=["Predição"], response_model=PredictionOutput)
def predict_sentiment(review: ReviewInput):
    if pipeline is None:
        return {"error": "Modelo não foi carregado."} 

    cleaned_text = clean_text(review.text)

    prediction = pipeline.predict([cleaned_text])
    label = int(prediction[0])
    sentiment = "positive" if label == 1 else "negative"

    return {"sentiment": sentiment, "label": label}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)