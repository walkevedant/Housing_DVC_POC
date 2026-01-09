import pandas as pd
import pickle
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

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

model = LinearRegression()
model.fit(X_train, y_train)

with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)
