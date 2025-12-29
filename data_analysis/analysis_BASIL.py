import os
import json
import pandas as pd
import numpy as np

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

def load_basil_articles(basil_dir):
    """
    load BASIL json files
    """
    articles = []
    for fname in os.listdir(basil_dir):
        if fname.endswith(".json"):
            path = os.path.join(basil_dir, fname)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                articles.append(data)
    return articles


def basil_to_sentences(articles):
    """
    convert BASIL articles to sentence-level records
    """
    rows = []

    for art_id, art in enumerate(articles):
        source = art.get("source", "")
        title = art.get("title", "")
        event = art.get("main-event", "")

        for sent in art["body"]:
            text = sent["sentence"]
            idx = sent["sentence-index"]
            annotations = sent.get("annotations", [])

            # binary label
            is_biased = 1 if len(annotations) > 0 else 0

            rows.append({
                "article_id": art_id,
                "source": source,
                "event": event,
                "title": title,
                "sentence_index": idx,
                "sentence": text,
                "biased": is_biased
            })

    return pd.DataFrame(rows)



def compute_sentiment(df):
    """
    VADER compound sentiment score
    """
    df["sentiment"] = df["sentence"].apply(
        lambda x: sia.polarity_scores(x)["compound"]
    )
    return df


def compute_article_level_contrast(df):
    """
    article-level mean sentiment and deviation
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
    local window sentiment contrast
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


def main():
    basil_dir = "../BASIL"

    print("Loading BASIL articles...")
    articles = load_basil_articles(basil_dir)
    print(f"Loaded {len(articles)} articles")

    print("Converting to sentence-level dataframe...")
    df = basil_to_sentences(articles)

    print("Computing sentiment...")
    df = compute_sentiment(df)

    print("Computing article-level contrast...")
    df = compute_article_level_contrast(df)

    print("Computing window-level contrast...")
    df = compute_window_contrast(df, window_size=3)

    print("\n=== Dataset Statistics ===")
    print(df["biased"].value_counts())

    print("\n=== Sentiment Mean ===")
    print(df.groupby("biased")["sentiment"].mean())

    print("\n=== Article-level Sentiment Deviation ===")
    print(df.groupby("biased")["sentiment_deviation_article"].mean())

    print("\n=== Window-level Sentiment Deviation ===")
    print(df.groupby("biased")["sentiment_deviation_window"].mean())

    # Save for later experiments
    df.to_csv("basil_sentiment_analysis.csv", index=False)
    print("\nSaved basil_sentiment_analysis.csv")


if __name__ == "__main__":
    main()