import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import precision_recall_fscore_support
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from tqdm import tqdm

# Dataset class to handle different input modes
class SentimentDataset(Dataset):
    def __init__(self, df, mode, tokenizer, max_length=256, window_size=2, top_k=2):
        """
        df: DataFrame containing the data for one fold
        mode: 'sentence', 'naive', 'contrastive-max', 'contrastive-min', or 'random'
        tokenizer: BERT tokenizer
        max_length: max token length for BERT inputs
        window_size: fixed window size for naive mode (±k sentences)
        top_k: number of contrastive sentences for contrastive modes
        """
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.window_size = window_size
        self.top_k = top_k

        # Group by article for context retrieval with sorted and reset index
        self.articles = {
            article_id: group
                .sort_values('sentence_index')
                .reset_index(drop=True)
            for article_id, group in self.df.groupby('article_id')
        }

        # Precompute sentiment scores per article for contrastive modes
        if mode in ['contrastive-max', 'contrastive-min']:
            # For each article, get sentiment scores as numpy array for fast access
            self.sentiment_scores = {}
            for article_id, group in self.articles.items():
                self.sentiment_scores[article_id] = group['sentiment'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        article_id = row['article_id']
        target_idx = row['sentence_index']
        label = int(row['biased'])

        # Sentence text
        target_sentence = row['sentence']

        if self.mode == 'sentence':
            # Only target sentence
            text = target_sentence

        elif self.mode == 'naive':
            # Fixed window ±k sentences around target in the same article
            group = self.articles[article_id]
            assert group.iloc[target_idx]['sentence_index'] == target_idx, \
                f"Sentence index mismatch in article {article_id}"
            start = max(target_idx - self.window_size, 0)
            end = min(target_idx + self.window_size + 1, len(group))
            context_sents = group.iloc[start:end]['sentence'].tolist()
            text = " ".join(context_sents)

        elif self.mode in ['contrastive-max', 'contrastive-min', 'random']:
            group = self.articles[article_id]
            assert group.iloc[target_idx]['sentence_index'] == target_idx, \
                f"Sentence index mismatch in article {article_id}"

            if self.mode == 'random':
                candidates = group['sentence_index'].tolist()
                candidates.remove(target_idx)
                top_indices = np.random.choice(
                    candidates,
                    size=min(self.top_k, len(candidates)),
                    replace=False
                )
            else:
                target_score = self.sentiment_scores[article_id][target_idx]
                diffs = np.abs(self.sentiment_scores[article_id] - target_score)
                diffs[target_idx] = np.inf

                if self.mode == 'contrastive-max':
                    top_indices = diffs.argsort()[-self.top_k:]
                else:  # contrastive-min
                    top_indices = diffs.argsort()[:self.top_k]

            top_indices = np.sort(top_indices)

            contrastive_sents = (
                group[group['sentence_index'].isin(top_indices)]
                .sort_values('sentence_index')['sentence']
                .tolist()
            )

            text = target_sentence + " [SEP] " + " [SEP] ".join(contrastive_sents)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        # Return input_ids, attention_mask and label
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    losses = []
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)

def eval_model(model, dataloader, device, eval_df, save_path=None):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            preds.extend(predictions.cpu().numpy())
            trues.extend(labels.cpu().numpy())
    precision, recall, f1, _ = precision_recall_fscore_support(trues, preds, pos_label=1, average='binary')

    errors = []
    for i, (y_true, y_pred) in enumerate(zip(trues, preds)):
        if y_true != y_pred:
            row = eval_df.iloc[i]
            errors.append({
                'article_id': row['article_id'],
                'sentence_index': row['sentence_index'],
                'sentence': row['sentence'],
                'sentiment': row['sentiment'],
                'gold_label': y_true,
                'pred_label': y_pred
            })

    if save_path is not None and len(errors) > 0:
        pd.DataFrame(errors).to_csv(save_path, index=False)

    return precision, recall, f1

def main():
    parser = argparse.ArgumentParser(description="Contrastive Sentiment Context Bias Detection Training")
    parser.add_argument('--data_path', type=str, required=True,
                        help="Path to preprocessed CSV file")
    parser.add_argument('--mode', type=str, choices=['sentence', 'naive', 'contrastive-max', 'contrastive-min', 'random'], default='sentence',
                        help="Input mode: sentence, naive, contrastive-max, contrastive-min, or random")
    parser.add_argument('--epochs', type=int, default=3, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size")
    parser.add_argument('--lr', type=float, default=2e-5, help="Learning rate")
    parser.add_argument('--kfold', type=int, default=5, help="Number of folds for cross-validation")
    parser.add_argument('--window_size', type=int, default=2, help="Window size for naive mode")
    parser.add_argument('--top_k', type=int, default=2, help="Number of top contrastive sentences for contrastive modes")
    parser.add_argument('--max_length', type=int, default=256, help="Max token length for BERT inputs")
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.data_path)

    # We expect the dataframe to have columns:
    # 'article_id' (int or str): unique article identifier
    # 'sentence_index' (int): sentence index within article (0-based)
    # 'sentence' (str): text of the sentence
    # 'sentiment' (float): sentiment score of the sentence (e.g., from -1 to 1)
    # 'biased' (int): 0/1 label for bias presence

    # Ensure required columns exist
    required_cols = ['article_id', 'sentence_index', 'sentence', 'sentiment', 'biased']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in data")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    gkf = GroupKFold(n_splits=args.kfold)
    articles = df['article_id']

    precisions = []
    recalls = []
    f1s = []

    print(f"Starting {args.kfold}-fold cross-validation with mode={args.mode}")

    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, groups=articles)):
        print(f"\nFold {fold+1}/{args.kfold}")

        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        train_dataset = SentimentDataset(train_df, args.mode, tokenizer,
                                         max_length=args.max_length,
                                         window_size=args.window_size,
                                         top_k=args.top_k)
        val_dataset = SentimentDataset(val_df, args.mode, tokenizer,
                                       max_length=args.max_length,
                                       window_size=args.window_size,
                                       top_k=args.top_k)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
        model.to(device)

        optimizer = AdamW(model.parameters(), lr=args.lr)

        for epoch in range(args.epochs):
            print(f"Epoch {epoch+1}/{args.epochs}")
            train_loss = train_epoch(model, train_loader, optimizer, device)
            print(f"Train loss: {train_loss:.4f}")

        error_path = f"errors_fold{fold+1}_{args.mode}.csv"
        precision, recall, f1 = eval_model(
            model,
            val_loader,
            device,
            val_df.reset_index(drop=True),
            save_path=error_path
        )
        print(f"Validation Precision (biased class): {precision:.4f}")
        print(f"Validation Recall (biased class): {recall:.4f}")
        print(f"Validation F1 (biased class): {f1:.4f}")

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    print("\nCross-validation results:")
    print(f"Average Precision: {np.mean(precisions):.4f}")
    print(f"Average Recall: {np.mean(recalls):.4f}")
    print(f"Average F1: {np.mean(f1s):.4f}")

if __name__ == "__main__":
    main()
