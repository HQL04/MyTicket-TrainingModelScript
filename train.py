import os
import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
from datasets                import load_dataset
from huggingface_hub         import HfApi
from sklearn.model_selection import GroupKFold
from sklearn                 import __version__ as sklearn_version
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType

print("Python version:", sys.version)
print("pandas:", pd.__version__)
print("numpy:", np.__version__)
print("lightgbm:", lgb.__version__)
print("scikit-learn:", sklearn.__version__)
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
# 8. Upload HuggingFace
# =========================
print("Uploading model to HuggingFace...")

api = HfApi()

api.upload_file(
    path_or_fileobj="event_ranker.onnx",
    path_in_repo="event_ranker.onnx",
    repo_id="HQL04/EventRecommendation",
    repo_type="model",
    token=HF_TOKEN
)

print("Model uploaded successfully!")
