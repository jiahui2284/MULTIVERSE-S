import os
import logging
from typing import List, Dict, Optional

import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import sqlite3
from datetime import datetime

import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

BASE_URL = "https://finance.yahoo.com"
NEWS_URL = f"{BASE_URL}/quote/EURUSD%3DX/latest-news/"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

CURRENCY_KEYWORDS = {
    "eurusd", "euro", "currency", "forex", "fx", "ecb", "dollar", "yen"
}

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _download_nltk_resources() -> None:
    """Download required NLTK resources if missing."""
    try:
        stopwords.words("english")
    except LookupError:
        nltk.download("stopwords", quiet=True)
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)


def scrape_news(limit: int = 10, timeout: int = 10) -> List[Dict[str, str]]:
    articles: List[Dict[str, str]] = []
    try:
        resp = requests.get(NEWS_URL, headers=HEADERS, timeout=timeout)
        if resp.status_code != 200:
            logging.error(f"GET {NEWS_URL} status {resp.status_code}")
            return articles

        soup = BeautifulSoup(resp.text, "html.parser")
        items = soup.select("li.js-stream-content") or soup.select("li[data-type='story']")
        if items:
            logging.info(f"Found {len(items)} stream items on listing page.")
            for item in items[:limit]:
                h3 = item.find("h3")
                a = item.find("a", href=True)
                if not a:
                    continue
                headline = h3.get_text(strip=True) if h3 else a.get_text(strip=True)
                href = a["href"]
                link = href if href.startswith("http") else BASE_URL + href
                try:
                    article_resp = requests.get(link, headers=HEADERS, timeout=timeout)
                    if article_resp.status_code != 200:
                        logging.warning(f"GET article {link} status {article_resp.status_code}")
                        continue
                    article_soup = BeautifulSoup(article_resp.text, "html.parser")
                    body = article_soup.select_one("div.caas-body")
                    text = ""
                    if body:
                        paragraphs = [p.get_text(" ", strip=True) for p in body.select("p")]
                        text = " ".join(paragraphs) if paragraphs else body.get_text(" ", strip=True)
                    if not text:
                        article_tag = article_soup.find("article")
                        if article_tag:
                            text = article_tag.get_text(" ", strip=True)
                    if not text:
                        logging.warning(f"No body text for article: {link}")
                        continue
                    blob = f"{headline} {text}".lower()
                    if not any(k in blob for k in CURRENCY_KEYWORDS):
                        continue
                    articles.append({"headline": headline, "text": text, "link": link})
                except requests.RequestException as e:
                    logging.warning(f"Request failed for {link}: {e}")
                    continue
        else:
            logging.warning("No structured stream items found; falling back to anchor scan.")
            links_seen = set()
            anchors = soup.find_all("a", href=True)
            for a in anchors:
                href = a["href"]
                if not href:
                    continue
                if "/news/" in href:
                    if href.startswith("http"):
                        parsed = urlparse(href)
                        if "finance.yahoo.com" not in (parsed.netloc or ""):
                            continue
                        link = href
                    else:
                        link = BASE_URL + href
                    if link in links_seen:
                        continue
                    links_seen.add(link)

                    # improved fallback headline
                    headline = a.get_text(strip=True)
                    if not headline:
                        slug = os.path.basename(link).replace("-", " ").replace(".html", "").title()
                        headline = slug or "(no headline)"

                    try:
                        article_resp = requests.get(link, headers=HEADERS, timeout=timeout)
                        if article_resp.status_code != 200:
                            logging.warning(f"GET article {link} status {article_resp.status_code}")
                            continue
                        article_soup = BeautifulSoup(article_resp.text, "html.parser")
                        body = article_soup.select_one("div.caas-body")
                        text = ""
                        if body:
                            paragraphs = [p.get_text(" ", strip=True) for p in body.select("p")]
                            text = " ".join(paragraphs) if paragraphs else body.get_text(" ", strip=True)
                        if not text:
                            article_tag = article_soup.find("article")
                            if article_tag:
                                text = article_tag.get_text(" ", strip=True)
                        if not text:
                            continue
                        blob = f"{headline} {text}".lower()
                        if not any(k in blob for k in CURRENCY_KEYWORDS):
                            continue
                        articles.append({"headline": headline, "text": text, "link": link})
                        if len(articles) >= limit:
                            break
                    except requests.RequestException:
                        continue
    except requests.RequestException as e:
        logging.error(f"Listing page request failed: {e}")
    return articles


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


def label_from_score(score: float) -> str:
    if score >= 0.05:
        return "positive"
    if score <= -0.05:
        return "negative"
    return "neutral"


def normalize_link(link: Optional[str]) -> Optional[str]:
    if not link:
        return link
    if link.endswith(".html/"):
        return link[:-1]
    return link


def init_db(db_path: str) -> None:
    """Create database and table if not exists, auto-create directory."""
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                headline TEXT,
                text TEXT,
                clean_text TEXT,
                sentiment REAL,
                sentiment_label TEXT,
                link TEXT,
                fetched_at TEXT,
                UNIQUE(link)
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def save_articles_to_db(
    articles: List[Dict[str, str]],
    db_path: str = "Project/news.db",
    symbol: str = "EURUSD=X",
) -> int:
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    inserted = 0
    try:
        cur = conn.cursor()
        for a in articles:
            link = normalize_link(a.get("link"))
            cur.execute(
                """
                INSERT OR IGNORE INTO articles (symbol, headline, text, clean_text, sentiment, sentiment_label, link, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol,
                    a.get("headline"),
                    a.get("text"),
                    a.get("clean_text"),
                    float(a.get("sentiment", 0.0)),
                    a.get("sentiment_label"),
                    link,
                    datetime.utcnow().isoformat(timespec="seconds") + "Z",
                ),
            )
            if cur.rowcount > 0:
                inserted += 1
        conn.commit()
        return inserted
    finally:
        conn.close()


def run_pipeline() -> List[Dict[str, str]]:
    articles = scrape_news(limit=10)
    if not articles:
        logging.info("Using sample articles for testing.")
        articles = [
            {
                "headline": "Dollar falls against euro as rate outlook shifts",
                "text": "The dollar weakened against the euro as traders bet on rate cuts and easing inflation.",
            },
            {
                "headline": "Euro slides amid growth concerns in the eurozone",
                "text": "The euro declined due to weak economic data and political uncertainty in Europe.",
            },
        ]
    for a in articles:
        clean = process_text(a["text"])
        score = analyze_sentiment(clean)
        a["clean_text"] = clean
        a["sentiment"] = score
        a["sentiment_label"] = label_from_score(score)
        a["link"] = normalize_link(a.get("link"))
    for idx, a in enumerate(articles, start=1):
        print(f"{idx}. {a['headline']}")
        if "link" in a:
            print(f"   Link: {a['link']}")
        print(f"   Sentiment: {a['sentiment']:.3f} ({a['sentiment_label']})")
    return articles


if __name__ == "__main__":
    results = run_pipeline()
    inserted = save_articles_to_db(results)
    print(f"Saved {inserted} new article(s) to database.") 