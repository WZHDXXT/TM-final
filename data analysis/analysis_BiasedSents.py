import os
import json
import pandas as pd
import numpy as np

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# =========================
# 0. Setup
# =========================

nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()


# =========================
# 1. Load BiasedSents JSON files
# =========================

def load_biasedsents_articles(biasedsents_dir):
    """
    Load all BiasedSents json files from a directory.
    Each file corresponds to one article.
    """
    articles = []
    for fname in os.listdir(biasedsents_dir):
        if fname.endswith(".json"):
            path = os.path.join(biasedsents_dir, fname)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                articles.append(data)
    return articles


# =========================
# 2. Convert to sentence-level DataFrame
# =========================

def biasedsents_to_sentences(articles):
    """
    Convert BiasedSents articles to sentence-level records.
    """
    rows = []

    for art_id, art in enumerate(articles):
        event = art.get("event", "")
        source = art.get("source", "")
        source_bias = art.get("source_bias", "")
        ref = art.get("ref", "")
        title = art.get("title", "")

        for sent in art["body"]:
            text = sent["sentence"]
            idx = sent["sentence_index"]
            ann = sent.get("ann", 0)

            rows.append({
                "article_id": art_id,
                "event": event,
                "source": source,
                "source_bias": source_bias,
                "ref": ref,
                "title": title,
                "sentence_index": idx,
                "sentence": text,
                "biased": ann
            })

    return pd.DataFrame(rows)


# =========================
# 3. Sentiment computation
# =========================

def compute_sentiment(df):
    """
    Add VADER compound sentiment score.
    """
    df["sentiment"] = df["sentence"].apply(
        lambda x: sia.polarity_scores(x)["compound"]
    )
    return df


# =========================
# 4. Contrast features
# =========================

def compute_article_level_contrast(df):
    """
    Compute article-level mean sentiment and deviation.
    """
    article_mean = (
        df.groupby("article_id")["sentiment"]
        .mean()
        .rename("article_mean_sentiment")
    )

    df = df.merge(article_mean, on="article_id")
    df["sentiment_deviation_article"] = (
        df["sentiment"] - df["article_mean_sentiment"]
    ).abs()

    return df


def compute_window_contrast(df, window_size=3):
    """
    Compute local window sentiment contrast.
    """
    df = df.sort_values(["article_id", "sentence_index"])
    window_devs = []

    for art_id, group in df.groupby("article_id"):
        sentiments = group["sentiment"].values
        indices = group.index.tolist()

        for i, idx in enumerate(indices):
            start = max(0, i - window_size)
            end = min(len(sentiments), i + window_size + 1)

            window = sentiments[start:end]
            window_mean = window.mean()

            window_devs.append(
                abs(sentiments[i] - window_mean)
            )

    df["sentiment_deviation_window"] = window_devs
    return df


# =========================
# 5. Main
# =========================

def main():
    biasedsents_dir = "./BiasedSents"  # 修改为你的 BiasedSents 路径

    print("Loading BiasedSents articles...")
    articles = load_biasedsents_articles(biasedsents_dir)
    print(f"Loaded {len(articles)} articles")

    print("Converting to sentence-level dataframe...")
    df = biasedsents_to_sentences(articles)

    print("Computing sentiment...")
    df = compute_sentiment(df)

    print("Computing article-level contrast...")
    df = compute_article_level_contrast(df)

    print("Computing window-level contrast...")
    df = compute_window_contrast(df, window_size=3)

    # =========================
    # 6. Basic statistics output
    # =========================

    print("\n=== Dataset Statistics ===")
    print(df["biased"].value_counts())

    print("\n=== Sentiment Mean ===")
    print(df.groupby("biased")["sentiment"].mean())

    print("\n=== Article-level Sentiment Deviation ===")
    print(df.groupby("biased")["sentiment_deviation_article"].mean())

    print("\n=== Window-level Sentiment Deviation ===")
    print(df.groupby("biased")["sentiment_deviation_window"].mean())

    # Save for later experiments
    df.to_csv("biasedsents_sentiment_analysis.csv", index=False)
    print("\nSaved biasedsents_sentiment_analysis.csv")


if __name__ == "__main__":
    main()