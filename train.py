import os
import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
import datasets
import huggingface_hub
import onnxmltools
from datasets import load_dataset
from huggingface_hub import HfApi
from sklearn.model_selection import GroupKFold
from sklearn import __version__ as sklearn_version
from onnxmltools.convert.common.data_types import FloatTensorType

import json
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

print("Python:", sys.version)
print("pandas:", pd.__version__)
print("numpy:", np.__version__)
print("lightgbm:", lgb.__version__)
print("sklearn:", sklearn_version)
print("onnxmltools:", onnxmltools.__version__)
print("datasets:", datasets.__version__)
print("huggingface_hub:", huggingface_hub.__version__)

HF_TOKEN = os.environ["HF_TOKEN"]

print("Loading dataset from HuggingFace...")

dataset = load_dataset(
    "HQL04/MyTicket-training-dataset",
    token=HF_TOKEN
)

df = dataset["train"].to_pandas()

# =========================
# 1. Normalize label
# =========================
def normalize_score(score):
    if score == 0: return 0
    elif score <= 5: return 1
    elif score <= 20: return 2
    elif score <= 60: return 3
    else: return 4

df["interestScore"] = df["interestScore"].apply(normalize_score)

print("Label distribution:")
print(df["interestScore"].value_counts())

# =========================
# 2. Features
# =========================
features = [
    "age",
    "gender",
    "avgPurchasePrice",
    "totalSpent",
    "totalTicketsPurchase",
    "genre",
    "ageLimit",
    "eventClickCount",
    "totalTickets",
    "sameEventGenreClickCount",
    "sameEventGenrePurchase"
]

target = "interestScore"

# ⚠️ QUAN TRỌNG: encode category -> số (tránh lỗi ONNX)
df["gender"] = df["gender"].astype("category").cat.codes
df["genre"]  = df["genre"].astype("category").cat.codes

# sort theo user để group đúng
df = df.sort_values("user_id")

X = df[features].astype("float32")   # ⚠️ ép float32 cho ONNX
y = df[target]
groups = df["user_id"]

# =========================
# 3. Params
# =========================
params = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_at": [15],
    "learning_rate": 0.03,
    "num_leaves": 64,
    "label_gain": [0, 1, 3, 7, 15],
    "verbosity": -1
}

# =========================
# 4. 5-Fold CV (chỉ để tìm best iteration)
# =========================
print("Starting Group K-Fold training...")

gkf = GroupKFold(n_splits=5)
best_iterations = []
scores = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
    print(f"\nTraining fold {fold+1}")

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val     = X.iloc[val_idx], y.iloc[val_idx]

    train_users = groups.iloc[train_idx]
    val_users   = groups.iloc[val_idx]

    train_group = train_users.groupby(train_users).size().to_list()
    val_group   = val_users.groupby(val_users).size().to_list()

    train_data = lgb.Dataset(X_train, label=y_train, group=train_group)
    val_data   = lgb.Dataset(X_val, label=y_val, group=val_group)

    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=2000,
        callbacks=[lgb.early_stopping(50)]
    )

    best_iterations.append(model.best_iteration)
    scores.append(model.best_score["valid_0"]["ndcg@15"])

print("\nKFold finished")
print("Mean NDCG:", np.mean(scores))

best_round = int(np.mean(best_iterations))
print("Best iteration:", best_round)

# =========================
# Precision@K & Recall@K
# =========================
def precision_recall_at_k(y_true, y_scores, k=10):

    top_k_idx = np.argsort(y_scores)[::-1][:k]

    y_true_topk = y_true.iloc[top_k_idx]

    relevant = (y_true_topk >= 3).sum()

    precision = relevant / k

    total_relevant = (y_true >= 3).sum()

    recall = relevant / total_relevant if total_relevant > 0 else 0

    return precision, recall

precision_scores = []
recall_scores = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val     = X.iloc[val_idx], y.iloc[val_idx]

    train_users = groups.iloc[train_idx]
    val_users   = groups.iloc[val_idx]

    train_group = train_users.groupby(train_users).size().to_list()

    train_data = lgb.Dataset(
        X_train,
        label=y_train,
        group=train_group
    )

    model = lgb.train(
        params,
        train_data,
        num_boost_round=best_round
    )

    y_pred = model.predict(X_val)

    precision, recall = precision_recall_at_k(
        y_val,
        y_pred,
        k=10
    )

    precision_scores.append(precision)
    recall_scores.append(recall)

mean_precision = float(np.mean(precision_scores))
mean_recall = float(np.mean(recall_scores))

print("Mean Precision@10:", mean_precision)
print("Mean Recall@10:", mean_recall)

# =========================
# MRR Evaluation
# =========================
print("\nCalculating MRR...")

def reciprocal_rank_per_user(y_true, y_scores):

    sorted_idx = np.argsort(y_scores)[::-1]

    y_true_sorted = y_true.iloc[sorted_idx]

    for rank, label in enumerate(y_true_sorted, start=1):

        if label >= 3:
            return 1 / rank

    return 0

mrr_scores = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val     = X.iloc[val_idx], y.iloc[val_idx]

    train_users = groups.iloc[train_idx]
    val_users   = groups.iloc[val_idx]

    train_group = train_users.groupby(train_users).size().to_list()

    train_data = lgb.Dataset(
        X_train,
        label=y_train,
        group=train_group
    )

    model = lgb.train(
        params,
        train_data,
        num_boost_round=best_round
    )

    y_pred = model.predict(X_val)

    val_df = pd.DataFrame({
        "user": val_users.values,
        "label": y_val.values,
        "score": y_pred
    })

    user_rr = []

    for user_id, group_df in val_df.groupby("user"):

        rr = reciprocal_rank_per_user(
            group_df["label"],
            group_df["score"]
        )

        user_rr.append(rr)

    mrr_scores.append(np.mean(user_rr))

mean_mrr = float(np.mean(mrr_scores))

print("Mean MRR:", mean_mrr)

# =========================
# 5. Train FINAL model (1 model duy nhất)
# =========================
best_round = int(np.mean(best_iterations))
print("Best iteration:", best_round)

print("\nTraining final model on full dataset...")

group_all = groups.groupby(groups).size().to_list()

train_data = lgb.Dataset(
    X,
    label=y,
    group=group_all
)

final_model = lgb.train(
    params,
    train_data,
    num_boost_round=best_round
)

# =========================
# 6. Save model
# =========================
print("Saving model...")
final_model.save_model("event_ranker.txt")

# =========================
# 7. Convert ONNX
# =========================
print("Converting to ONNX...")

initial_type = [("float_input", FloatTensorType([None, len(features)]))]

onnx_model = onnxmltools.convert_lightgbm(
    final_model,
    initial_types=initial_type
)

with open("event_ranker.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

# =========================
# Feature Importance
# =========================
print("Generating Feature Importance...")

feature_imp = pd.DataFrame(
    sorted(
        zip(final_model.feature_importance(), features)
    ),
    columns=['Value', 'Feature']
)

plt.figure(figsize=(10, 6))

sns.barplot(
    x="Value",
    y="Feature",
    data=feature_imp.sort_values(
        by="Value",
        ascending=False
    )
)

plt.title('Event Recommendation Feature Importance')

plt.tight_layout()

plt.savefig("feature_importance.png")

plt.close()

# =========================
# Confusion Matrix
# =========================
print("Generating Confusion Matrix...")

y_pred_score = final_model.predict(X)

y_pred_label = pd.qcut(
    y_pred_score,
    q=5,
    labels=[0,1,2,3,4]
)

report = classification_report(
    y,
    y_pred_label
)

with open("classification_report.txt", "w") as f:
    f.write(report)

cm = confusion_matrix(y, y_pred_label)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')

plt.tight_layout()

plt.savefig("confusion_matrix.png")

plt.close()

# =========================
# Save Metrics
# =========================
metrics = {
    "ndcg@15": float(np.mean(scores)),
    "precision@10": mean_precision,
    "recall@10": mean_recall,
    "mrr": mean_mrr,
    "best_iteration": int(best_round)
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Metrics saved!")

# =========================
# 8. Upload HuggingFace
# =========================
print("Uploading model to HuggingFace...")

api = HfApi()

files_to_upload = [
    "event_ranker.onnx",
    "feature_importance.png",
    "confusion_matrix.png",
    "classification_report.txt",
    "metrics.json"
]

for file in files_to_upload:

    print(f"Uploading {file}...")

    api.upload_file(
        path_or_fileobj=file,
        path_in_repo=file,
        repo_id="HQL04/EventRecommendation",
        repo_type="model",
        token=HF_TOKEN
    )

print("All artifacts uploaded successfully!")
