# Contrastive Sentiment Context for Sentence-Level Media Bias Detection

This project explores **sentence-level media bias detection** using a simple and interpretable **contrastive sentiment context** approach.  
We investigate whether **relative sentiment differences between sentences within the same article** can serve as an effective contextual signal for identifying biased framing.

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

- Two backbones supported via separate scripts: BERT-base and RoBERTa-base.
- Task: Binary classification (biased vs. unbiased)
- Optimizer: AdamW
- Evaluation metric: Precision, Recall, F1 (biased class)
- Validation: Article-level *k*-fold cross-validation (GroupKFold)

---

### Backbone Selection

The choice of encoder backbone is controlled by selecting the corresponding training script.  
Use `train_contrastive_BERT.py` for BERT-base experiments and `train_contrastive_RoBERTa.py` for RoBERTa-base experiments.  
All other experimental settings and arguments remain identical across both scripts.

---

### Setup

Before running any training commands, install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

Run training with different context selection strategies by specifying the `mode` argument.

```bash
# Sentence-only baseline (no context) using BERT
python train_contrastive_BERT.py \
  --data_path data_analysis/basil_sentiment_analysis.csv \
  --mode sentence

# Sentence-only baseline (no context) using RoBERTa
python train_contrastive_RoBERTa.py \
  --data_path data_analysis/basil_sentiment_analysis.csv \
  --mode sentence

# Naive window-based context using BERT
python train_contrastive_BERT.py \
  --data_path data_analysis/basil_sentiment_analysis.csv \
  --mode naive 

# Naive window-based context using RoBERTa
python train_contrastive_RoBERTa.py \
  --data_path data_analysis/basil_sentiment_analysis.csv \
  --mode naive 

# Contrastive sentiment context (maximum sentiment contrast) using BERT
python train_contrastive_BERT.py \
  --data_path data_analysis/basil_sentiment_analysis.csv \
  --mode contrastive \
  --top_k 2

# Contrastive sentiment context (maximum sentiment contrast) using RoBERTa
python train_contrastive_RoBERTa.py \
  --data_path data_analysis/basil_sentiment_analysis.csv \
  --mode contrastive \
  --top_k 2

# Randomly sampled context (control baseline) using BERT
python train_contrastive_BERT.py \
  --data_path data_analysis/basil_sentiment_analysis.csv \
  --mode random \
  --top_k 2

# Randomly sampled context (control baseline) using RoBERTa
python train_contrastive_RoBERTa.py \
  --data_path data_analysis/basil_sentiment_analysis.csv \
  --mode random \
  --top_k 2
```