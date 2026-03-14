import os
import pandas as pd
import lightgbm as lgb
from datasets import load_dataset
from huggingface_hub import HfApi
from sklearn.model_selection import train_test_split
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType

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

# Sort theo user_id để group đúng
df = df.sort_values("user_id")

X = df[features]
y = df[target]

print("Preparing ranking groups...")

# mỗi user là một query
group = df.groupby("user_id").size().to_list()

train_data = lgb.Dataset(
    X,
    label=y,
    group=group
)

params = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "learning_rate": 0.05,
    "num_leaves": 31
}

print("Training LightGBM ranker...")

model = lgb.train(
    params,
    train_data,
    num_boost_round=100
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