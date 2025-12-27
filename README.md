# Contrastive Sentiment Context for Sentence-Level Media Bias Detection

This project explores **sentence-level media bias detection** using a simple and interpretable **contrastive sentiment context** approach.  
We investigate whether **relative sentiment differences between sentences within the same article** can serve as an effective contextual signal for identifying biased framing.

---

## Motivation

Prior work on sentence-level media bias detection often relies on complex discourse structures, event graphs, or dataset-specific annotations.  
In contrast, this project focuses on a **lightweight, sentence-centric formulation**, asking whether minimal contextual information—captured through sentiment contrast—is sufficient for bias prediction.

---

## Data Preparation

Model inputs are generated through a **prior analysis step**, which computes sentence-level sentiment scores and attaches them to the dataset.

The model expects a CSV file with the following columns:

| Column | Description |
|------|-------------|
| `article_id` | Unique article identifier |
| `sentence_index` | Sentence position in the article (0-based, contiguous) |
| `sentence` | Sentence text |
| `sentiment` | Sentence-level sentiment score (analysis-generated) |
| `biased` | Binary bias label (0 = unbiased, 1 = biased) |

The `sentiment` field is an auxiliary signal (e.g., VADER or TextBlob output) and is **not** a gold label.  
It is used **only for selecting contextual sentences**, not for supervision.

Datasets such as **BASIL** and **BiasedSents** are first converted into this format before training.

---

## Context Selection Modes

The training script supports multiple context strategies:

- `sentence` — target sentence only  
- `naive` — fixed window of surrounding sentences  
- `contrastive-max` — top-k sentences with maximum sentiment contrast  
- `contrastive-min` — top-k sentences with minimum sentiment contrast  
- `random` — randomly sampled contextual sentences  

---

## Model and Training

- Backbone: `bert-base-uncased`
- Task: Binary classification (biased vs. unbiased)
- Optimizer: AdamW
- Evaluation metric: Precision, Recall, F1 (biased class)
- Validation: Article-level *k*-fold cross-validation (GroupKFold)

---

## Usage

Run training with different context selection strategies by specifying the `mode` argument.

```bash
# Sentence-only baseline (no context)
python train_contrastive.py \
  --data_path data/basil_sentiment_analysis.csv \
  --mode sentence

# Naive window-based context
python train_contrastive.py \
  --data_path data/basil_sentiment_analysis.csv \
  --mode naive \
  --window_size 2

# Contrastive sentiment context (maximum contrast)
python train_contrastive.py \
  --data_path data/basil_sentiment_analysis.csv \
  --mode contrastive-max \
  --top_k 2

# Contrastive sentiment context (minimum contrast)
python train_contrastive.py \
  --data_path data/basil_sentiment_analysis.csv \
  --mode contrastive-min \
  --top_k 2

# Randomly sampled context (control baseline)
python train_contrastive.py \
  --data_path data/basil_sentiment_analysis.csv \
  --mode random \
  --top_k 2
```

---

## Notes

- All context is retrieved **within the same article** to avoid data leakage.
- Sentence indices must be sorted and contiguous within each article.
- Error analysis outputs are saved automatically during evaluation.