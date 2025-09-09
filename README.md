Gen-AI & ML Based Stock Market Prediction

End-to-end pipeline that ingests market data, engineers technical & sentiment features, trains ML/Gen-AI models, and serves predictions via a simple app API/UI.
Focused on Indian equities (mid-cap friendly), but adaptable to any ticker/universe.

✨ Key Features

Data Ingestion

Historical OHLCV from brokers/APIs (e.g., Yahoo Finance) and local CSVs

Optional news & social sentiment (e.g., X/Twitter via snscrape)

Feature Engineering

Technical indicators (MA, RSI, MACD, ATR, Bollinger, etc.)

Price/volume transforms, lagged features, rolling stats

Optional NLP embeddings & sentiment scores (transformers)

Models

Classical ML: Linear/Logistic, RandomForest, XGBoost, LightGBM

Time-series: ARIMA/Prophet (optional)

Deep/NLP (optional): LSTM/Transformer for sequence & sentiment fusion

Backtesting & Metrics

Walk-forward splits, returns curves, precision/recall, MAPE, Sharpe

Serving

app.py for a lightweight REST/Streamlit app

Saved models + config for reproducible runs

Reproducibility

Deterministic seeds, environment files, Makefile tasks, CI-friendly layout

🗂️ Project Structure
Gen-AI-ML-based-stock-market-precdiction/
├─ app.py                       # Web/API entrypoint
├─ train_stock_model.py         # Main training script
├─ inference.py                 # Batch/online prediction
├─ twitter_sentiment_analysis.py# Sentiment pipeline (snscrape, transformers)
├─ rag.py                       # (Optional) RAG summarizer for news sentiment
├─ requirements.txt             # Python dependencies
├─ README.md                    # This file
├─ .env.example                 # Example environment variables
├─ data/
│  ├─ raw/                      # Unmodified source files
│  ├─ interim/                  # Cleaned/intermediate
│  └─ processed/                # Feature matrices, train/test splits
├─ models/
│  ├─ artifacts/                # Pickled models, tokenizers
│  └─ metrics/                  # JSON/CSV of evaluation metrics
├─ notebooks/
│  ├─ 01_eda.ipynb
│  ├─ 02_features.ipynb
│  └─ 03_modeling.ipynb
├─ src/
│  ├─ config.py                 # Paths, constants, hyperparams
│  ├─ data/
│  │  ├─ loaders.py             # yfinance/CSV ingestion
│  │  └─ preprocess.py          # cleaning, missing values
│  ├─ features/
│  │  ├─ technical.py           # indicators (ta)
│  │  └─ build_features.py
│  ├─ models/
│  │  ├─ ml.py                  # RF/XGB/LGBM wrappers
│  │  ├─ timeseries.py          # ARIMA/Prophet
│  │  └─ deep.py                # LSTM/Transformer (optional)
│  ├─ nlp/
│  │  ├─ sentiment.py           # snscrape + transformers
│  │  └─ embed.py               # text embeddings
│  ├─ backtest/
│  │  ├─ splitter.py            # time series splits
│  │  └─ evaluate.py            # metrics, plots
│  └─ utils/
│     ├─ io.py                  # save/load artifacts
│     ├─ logging.py
│     └─ seeds.py
└─ tests/
   └─ test_basic.py
🚀 Quickstart
1) Prerequisites

Python 3.10+

Git

(Optional) GPU + PyTorch for deep/NLP

2) Clone & Environment
git clone https://github.com/ADI14M/Gen-AI-ML-based-stock-market-precdiction.git
cd Gen-AI-ML-based-stock-market-precdiction

# Create & activate venv
python -m venv .venv
# Windows:
. .venv/Scripts/activate
# macOS/Linux:
# source .venv/bin/activate

# Install deps
pip install --upgrade pip
pip install -r requirements.txt

3) Configure Environment

Create .env from the template:
cp .env.example .env
# Data
DATA_DIR=./data
TICKER=RELIANCE.NS            # Example NSE ticker; use ^NSEI for index, or any Yahoo code
START_DATE=2016-01-01
END_DATE=2025-09-01
INTERVAL=1d

# Optional: sentiment/NLP
USE_SENTIMENT=false
MAX_TWEETS=500
OPENAI_API_KEY=
HUGGINGFACE_API_KEY=

4) Train a Model
python train_stock_model.py \
  --ticker "%TICKER%" \
  --start "%START_DATE%" \
  --end "%END_DATE%" \
  --interval "%INTERVAL%" \
  --model xgboost \
  --use-sentiment %USE_SENTIMENT%
5) Run App (UI/API)

# If Streamlit app:
streamlit run app.py

# If FastAPI/Flask app:
python app.py
# or
uvicorn app:app --reload --port 8000

⚙️ Configuration & Common Flags

Most scripts accept these flags (check --help on each):

--ticker (str): Yahoo Finance symbol (e.g., RELIANCE.NS, TCS.NS, HDFCBANK.NS)

--start, --end (YYYY-MM-DD)

--interval (e.g., 1d, 1h, 5m)

--horizon (int): Predict h steps ahead

--target (str): close, return, or classification label

--model (str): linear, rf, xgboost, lightgbm, prophet, lstm

--use-sentiment (bool): Include sentiment features

--save-artifacts (bool): Save model + scaler + feature config

🧪 Example Workflow

Ingest & Clean
python -m src.data.loaders --ticker RELIANCE.NS --start 2016-01-01 --end 2025-09-01
python -m src.data.preprocess

Build Features
python -m src.features.build_features --add-tech --lags 5 --rolls 5 10 20

📊 Metrics & Backtesting

Regression: RMSE, MAE, MAPE, R²

Classification: Accuracy, Precision/Recall, F1, ROC-AUC

Strategy: Cumulative returns, Max drawdown, Sharpe

Outputs saved under:

models/metrics/*.json

models/artifacts/*.*

data/processed/*.*
🧠 Sentiment & NLP (Optional)

Twitter/X: snscrape for recent tweets, cached to data/raw/tweets_{ticker}.jsonl

Transformers: zero-shot or finetuned sentiment models

Fusion: join daily sentiment aggregates (mean, std, pos_ratio) with technical features before training
Run: python twitter_sentiment_analysis.py --ticker RELIANCE.NS --max 500 --out data/processed/sentiment_REL.json

🧩 RAG Summary (Optional)

If you use rag.py for daily news summaries:

Avoid circular imports (do not from rag import generate_ai_summary inside rag.py).

Prefer:# app.py
from rag import generate_ai_summary
🧰 Requirements

Example requirements.txt (trim or expand based on your code):
pandas
numpy
scikit-learn
xgboost
lightgbm
ta
yfinance
matplotlib
plotly
snscrape
transformers
torch
tqdm
python-dotenv
prophet
statsmodels
requests
beautifulsoup4
streamlit
fastapi
uvicorn

🧪 Testing
pip install pytest
pytest -q

🔒 Notes & Disclaimers

This project is for research/education. It is not financial advice.

Past performance is not indicative of future results. Use at your own risk.

🗺️ Roadmap

 Unified config & CLI

 Model registry + experiment tracking (MLflow/Weights&Biases)

 Live paper trading (Broker API)

 Hyperparameter search (Optuna)

 Portfolio modeling (risk parity, mean-variance)

🤝 Contributing

PRs welcome!

Fork the repo

Create a feature branch

Add tests/docs

Open a PR
🙌 Acknowledgements

yfinance, ta, scikit-learn, xgboost, lightgbm, transformers, snscrape, prophet, streamlit, fastapi

TL;DR

pip install -r requirements.txt

Edit .env

python train_stock_model.py --model xgboost

streamlit run app.py
