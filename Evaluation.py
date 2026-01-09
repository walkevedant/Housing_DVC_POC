import pandas as pd
import pickle
import json
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import yaml

with open("params.yaml") as f:
    params = yaml.safe_load(f)

df = pd.read_csv("data/processed/data.csv")

X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=params["train"]["test_size"],
    random_state=params["train"]["random_state"]
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

y_pred = model.predict(X_test)

metrics = {
    "rmse": mean_squared_error(y_test, y_pred, squared=False),
    "r2": r2_score(y_test, y_pred)
}

with open("metrics/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)
