import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ========== 1. Load data ==========
df = pd.read_csv("basil_sentiment_analysis.csv")

# Convert bias indicator to readable labels (for paper presentation)
df["bias_label"] = df["biased"].map({1: "Biased", 0: "Non-biased"})
ORDER = ["Non-biased", "Biased"]

# Set unified, paper-friendly style
sns.set(style="whitegrid", font_scale=1.1)

# ========== 2. Create figure ==========
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=False)

# -------- (a) Sentiment Distribution --------
sns.violinplot(
    data=df,
    x="bias_label",
    y="sentiment",
    order=ORDER,    
    ax=axes[0],
    inner="box",
    cut=0
)
axes[0].set_title("(a) Sentiment Polarity")
axes[0].set_xlabel("")
axes[0].set_ylabel("VADER Sentiment Score")

# -------- (b) Article-level Deviation --------
sns.boxplot(
    data=df,
    x="bias_label",
    y="sentiment_deviation_article",
    order=ORDER,    
    ax=axes[1]
)
axes[1].set_title("(b) Article-level Deviation")
axes[1].set_xlabel("")
axes[1].set_ylabel("Absolute Deviation")

# -------- (c) Window-level Deviation --------
sns.violinplot(
    data=df,
    x="bias_label",
    y="sentiment_deviation_window",
    order=ORDER,    
    ax=axes[2],
    inner="box",
    cut=0
)
axes[2].set_title("(c) Local Context Deviation")
axes[2].set_xlabel("")
axes[2].set_ylabel("Absolute Deviation")

# ========== 3. Layout adjustment ==========
plt.tight_layout()
plt.show()