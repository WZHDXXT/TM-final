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

This analysis step is performed **offline** using dataset-specific preprocessing scripts:

- `analysis_BASIL.py` — processes the BASIL dataset  
- `analysis_BiasedSents.py` — processes the BiasedSents dataset  

The preprocessing scripts produce the following files:

- `basil_sentiment_analysis.csv`
- `biasedsents_sentiment_analysis.csv`

These CSV files are **generated in advance** and are directly consumed by the training script.  
No sentiment computation or contrast extraction is performed during model training.

The `sentiment` field is an auxiliary signal (e.g., VADER or TextBlob output) and is **not** a gold label.  
It is used **only for selecting contextual sentences**, not for supervision.

Datasets such as **BASIL** and **BiasedSents** are first converted into this format before training.

---

## Context Selection Modes

The training script supports multiple context strategies:

- `sentence` — target sentence only  
- `naive` — fixed window of surrounding sentences  
- `contrastive` — top-k sentences with maximum sentiment contrast  
- `random` — randomly sampled contextual sentences  

---

## Model and Training

- Backbone: `bert-base-uncased`
- Task: Binary classification (biased vs. unbiased)
- Optimizer: AdamW
- Evaluation metric: Precision, Recall, F1 (biased class)
- Validation: Article-level *k*-fold cross-validation (GroupKFold)

---

### Setup

Before running any training commands, install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run training with different context selection strategies by specifying the `mode` argument.

```bash
# Sentence-only baseline (no context)
python train_contrastive.py \
  --data_path data_analysis/basil_sentiment_analysis.csv \
  --mode sentence

# Naive window-based context
python train_contrastive.py \
  --data_path data_analysis/basil_sentiment_analysis.csv \
  --mode naive 

# Contrastive sentiment context (maximum sentiment contrast)
python train_contrastive.py \
  --data_path data_analysis/basil_sentiment_analysis.csv \
  --mode contrastive \
  --top_k 2

# Randomly sampled context (control baseline)
python train_contrastive.py \
  --data_path data_analysis/basil_sentiment_analysis.csv \
  --mode random \
  --top_k 2
```