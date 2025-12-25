# Contrastive Sentiment Context for Sentence-Level Media Bias Detection

This project explores **sentence-level media bias detection** using a simple and interpretable **contrastive sentiment context** approach.  
We investigate whether **relative sentiment differences between sentences within the same article** can serve as an effective contextual signal for identifying biased framing.

---

## Overview

Instead of modeling complex discourse or event structures, we:
- Treat each sentence as the primary unit of prediction
- Select contextual sentences based on **sentiment contrast**
- Compare against sentence-only and naive context baselines

All model inputs are **generated through a prior analysis step**, where sentence-level sentiment scores are computed and attached to the dataset.

---

## Input Data Format

The model expects a CSV file produced by a **data analysis /**, with the following columns:

| Column | Description |
|------|-------------|
| `article_id` | Unique article identifier |
| `sentence_index` | Sentence position in article (0-based) |
| `sentence` | Sentence text |
| `sentiment` | Sentence-level sentiment score (analysis-generated) |
| `biased` | Binary bias label (0/1) |

The `sentiment` field is **not a gold label**, but an auxiliary signal obtained from sentiment analysis (e.g., VADER or TextBlob), and is used **only for context selection**.

Datasets such as **BASIL** and **BiasedSents** are first analyzed and converted into this format before training.

---

## Context Modes

- `sentence`: target sentence only  
- `naive`: fixed window of surrounding sentences  
- `contrastive`: top-k sentences with highest sentiment contrast

---

## Training

- Model: `bert-base-uncased`
- Task: Binary classification (biased vs. unbiased)
- Evaluation: Precision, Recall, F1 (biased class)
- Validation: Article-level k-fold cross-validation

---

## Usage

```bash
# Sentence-only baseline
python train_contrastive.py --data_path basil_sentiment_analysis.csv --mode sentence

# Naive context
python train_contrastive.py --data_path biasedsents_sentiment_analysis.csv --mode naive --window_size 2

# Contrastive sentiment context (proposed)
python train_contrastive.py --data_path basil_sentiment_analysis.csv --mode contrastive --top_k 2