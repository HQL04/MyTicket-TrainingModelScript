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

dataset = load_dataset("HQL04/MyTicket-training-dataset")
df = dataset["train"].to_pandas()

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

X = df[features]
y = df[target]

print("Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

train_data = lgb.Dataset(X_train, label=y_train)

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