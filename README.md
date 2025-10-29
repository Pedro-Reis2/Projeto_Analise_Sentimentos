# Projeto: API de Análise de Sentimento com FastAPI e Docker

Este projeto é uma API de Machine Learning "pronta para produção" que classifica o sentimento (positivo/negativo) de reviews de filmes.

## Visão Geral

O objetivo foi demonstrar o ciclo de vida completo de um produto de I.A.:

1.  **Treinamento:** Um modelo de Regressão Logística foi treinado no dataset de 50k reviews do IMDb, usando um pipeline `TfidfVectorizer` (ver `notebooks/treinamento.ipynb`).
2.  **Serviço:** A API foi construída com **FastAPI**, recebendo um texto JSON e retornando uma predição.
3.  **Containerização:** A aplicação é empacotada com **Docker**, pronta para deploy em qualquer ambiente de nuvem.

## Estrutura do Repositório

/seu-projeto/
├── app/                  # Código fonte da API
│   ├── main.py           # Lógica da API (FastAPI)
│   ├── Dockerfile        # Instruções para o contêiner
│   └── requirements.txt  # Dependências da API
├── data/
│   └── IMDB Dataset.csv  # (Ou um .txt dizendo "Baixar de... ")
├── models/
│   └── sentiment_pipeline.pkl # O modelo treinado
├── notebooks/
│   └── treinamento.ipynb # Notebook de exploração e treinamento
└── README.md             # Isso que você está lendo

## Como Rodar (Usando Docker - Recomendado)

Este é o método preferido, pois simula um ambiente de produção.

**Pré-requisitos:** [Docker](https://www.docker.com/get-started) instalado.

1.  **Clone o repositório:**
    ```bash
    git clone [URL-DO-SEU-GITHUB]
    cd seu-projeto
    ```

2.  **Construa a imagem Docker:**
    (A partir da pasta raiz, aponte para o Dockerfile dentro de /app)
    ```bash
    docker build -t sentiment-api -f app/Dockerfile .
    ```

3.  **Rode o contêiner:**
    ```bash
    docker run -p 8000:8000 sentiment-api
    ```

4.  **Teste!**
    Sua API agora está rodando. Abra seu navegador em:
    
    **`http://localhost:8000/docs`**
    
    Você verá a documentação automática do FastAPI, onde pode testar a API diretamente.

## A Conexão com a AWS (Estratégia de Deploy)

Este projeto está 100% pronto para a AWS. O próximo passo (que não foi executado aqui) seria:

1.  **AWS ECR (Elastic Container Registry):** Fazer o *push* da imagem Docker (`sentiment-api`) para um registro privado da AWS.
2.  **AWS App Runner (Opção Fácil):** Criar um serviço App Runner que aponta para a imagem no ECR. A AWS cuida de tudo.
3.  **AWS ECS/Fargate (Opção Robusta):** Criar um Cluster ECS e uma Task Definition para rodar o contêiner de forma escalável.