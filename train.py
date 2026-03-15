import  os
import  pandas      as pd
import  lightgbm    as lgb
from    datasets                import load_dataset
from    huggingface_hub         import HfApi
from    sklearn.model_selection import GroupKFold
import  onnxmltools
from    onnxmltools.convert.common.data_types import FloatTensorType

HF_TOKEN = os.environ["HF_TOKEN"]

print("Loading dataset from HuggingFace...")

dataset = load_dataset(
    "HQL04/MyTicket-training-dataset",
    token=HF_TOKEN
)

df = dataset["train"].to_pandas()

print("Normalizing interestScore...")

def normalize_score(score):
    if score == 0:
        return 0
    elif score <= 5:
        return 1
    elif score <= 20:
        return 2
    elif score <= 60:
        return 3
    else:
        return 4

df["interestScore"] = df["interestScore"].apply(normalize_score)

print("Label distribution:")
print(df["interestScore"].value_counts())

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

df = df.sort_values("user_id")

X = df[features]
y = df[target]
groups = df["user_id"]

params = {
    "objective": "lambdarank",
    "metric": "ndcg@15",
    "learning_rate": 0.03,
    "num_leaves": 31
}

print("Starting Group K-Fold training...")

gkf = GroupKFold(n_splits=5)

best_iterations = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):

    print(f"\nTraining fold {fold+1}")

    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_val = X.iloc[val_idx]
    y_val = y.iloc[val_idx]

    train_users = df.iloc[train_idx]["user_id"]
    val_users = df.iloc[val_idx]["user_id"]

    train_group = train_users.groupby(train_users).size().to_list()
    val_group = val_users.groupby(val_users).size().to_list()

    train_data = lgb.Dataset(
        X_train,
        label=y_train,
        group=train_group
    )

    val_data = lgb.Dataset(
        X_val,
        label=y_val,
        group=val_group
    )

    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(50)]
    )

    best_iterations.append(model.best_iteration)

print("\nKFold finished")

best_round = int(sum(best_iterations) / len(best_iterations))
print("Best iteration:", best_round)

print("\nTraining final model on full dataset...")

group = df.groupby("user_id").size().to_list()

train_data = lgb.Dataset(
    X,
    label=y,
    group=group
)

model = lgb.train(
    params,
    train_data,
    num_boost_round=best_round
)

print("Saving model...")

model.save_model("event_ranker.txt")

print("Converting to ONNX...")

initial_type = [("float_input", FloatTensorType([None, len(features)]))]

onnx_model = onnxmltools.convert_lightgbm(
    model,
    initial_types=initial_type
)

with open("event_ranker.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

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
