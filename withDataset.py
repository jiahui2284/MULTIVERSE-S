import os
import logging
import feedparser
from datetime import datetime
from typing import List, Dict

import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim

# ---------------- SETTINGS ----------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Yahoo Finance RSS feed for EUR/USD news
RSS_URL = "https://feeds.finance.yahoo.com/rss/2.0/headline?s=EURUSD=X&region=US&lang=en-US"

# Keywords to filter relevant news
CURRENCY_KEYWORDS = {"eurusd", "euro", "forex", "currency", "dollar", "ecb", "fx", "europe"}

# ---------------- NLTK INIT ----------------
def _download_nltk_resources():
    try:
        stopwords.words("english")
    except LookupError:
        nltk.download("stopwords", quiet=True)
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)

_download_nltk_resources()
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
analyzer = SentimentIntensityAnalyzer()

def process_text(text: str) -> str:
    lower = text.lower()
    no_punct = lower.translate(str.maketrans("", "", string.punctuation))
    tokens = [lemmatizer.lemmatize(w) for w in no_punct.split() if w not in stop_words]
    return " ".join(tokens)

def analyze_sentiment(text: str) -> float:
    return analyzer.polarity_scores(text).get("compound", 0.0)

# ---------------- FETCH NEWS ----------------
def fetch_yahoo_rss(limit: int = 10) -> List[Dict[str, str]]:
    """Fetch latest EUR/USD related news from Yahoo Finance RSS feed."""
    logging.info("Fetching Yahoo Finance RSS feed for EUR/USD...")
    feed = feedparser.parse(RSS_URL)
    articles = []

    for entry in feed.entries[:limit * 3]   :  # Fetch extra to filter later
        title = entry.get("title", "")
        summary = entry.get("summary", "")
        link = entry.get("link", "")
        pub_date = entry.get("published", "")
        combined = f"{title} {summary}".lower()

        # 只保留和 EUR/USD 相关的新闻
        if not any(k in combined for k in CURRENCY_KEYWORDS):
            continue

        clean = process_text(title + " " + summary)
        sentiment = analyze_sentiment(clean)
        articles.append({
            "headline": title,
            "summary": summary,
            "sentiment": sentiment,
            "link": link,
            "date": pub_date
        })

        if len(articles) >= limit:
            break

    logging.info(f"✅ Fetched {len(articles)} Yahoo Finance RSS news items.")
    for i, a in enumerate(articles, 1):
        print(f"{i}. {a['headline']}\n   Sentiment: {a['sentiment']:.3f}\n   Link: {a['link']}\n")
    return articles

# ---------------- LSTM MODEL ----------------
class LSTMModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=50, num_layers=1, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ---------------- PIPELINE ----------------
def run_pipeline():
    # Get historical EUR/USD data
    df = yf.download("USDEUR=X", start="2025-01-01", interval="1d")
    df = df.reset_index()[["Date", "Close"]].rename(columns={"Close": "Price"}).sort_values("Date")

    # Get latest news and average sentiment
    news = fetch_yahoo_rss(limit=10)
    avg_sentiment = np.mean([n["sentiment"] for n in news]) if news else 0.0
    logging.info(f"Average sentiment: {avg_sentiment:.3f}")

    # Normalize and prepare dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(df["Price"].values.reshape(-1, 1))

    sentiment_series = np.full_like(scaled_prices, avg_sentiment)
    combined = np.hstack([scaled_prices, sentiment_series])

    X, y = [], []
    for i in range(7, len(combined)):
        X.append(combined[i-7:i, :])
        y.append(combined[i, 0])

    X, y = np.array(X), np.array(y)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

    # Train LSTM model
    model = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(20):
        model.train()
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/10, Loss: {loss.item():.6f}")

    # Predict next day's exchange rate
    last_7 = combined[-7:]
    last_7 = torch.tensor(last_7, dtype=torch.float32).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        pred_scaled = model(last_7).numpy()

    pred_price = scaler.inverse_transform(pred_scaled)
    print(f"\n💹 Predicted next-day EUR/USD exchange rate: {float(pred_price[0][0]):.5f}")
    print(f"🧠 Based on average sentiment {avg_sentiment:.3f} ({'positive' if avg_sentiment>0 else 'negative' if avg_sentiment<0 else 'neutral'})")

if __name__ == "__main__":
    run_pipeline()
