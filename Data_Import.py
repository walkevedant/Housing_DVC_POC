import pandas as pd
import os

os.makedirs("data/processed", exist_ok=True)

df = pd.read_csv("california_housing.csv")

# Basic cleaning
df = df.fillna(df.mean())

# Feature engineering
#df["Rooms_per_Household"] = df["AveRooms"] / df["HouseAge"]
#df["Bedrooms_per_Room"] = df["AveBedrms"] / df["AveRooms"]

df.to_csv("data/processed/data.csv", index=False)
