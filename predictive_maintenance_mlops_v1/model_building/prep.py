# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/Rajanan/ds-predictive-engine-maintenance-v1/engine_data.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

print(df.columns.tolist())

df["pressure_diff"] = df["Fuel pressure"] - df["Coolant pressure"]
df["temp_diff"] = df["Coolant temp"] - df["lub oil temp"]

df["rpm_pressure_ratio"] = df["Engine rpm"] / (df["Fuel pressure"] + 1e-6)

target_col = "Engine Condition"

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

#Remove misleading features
X = X.drop(columns=[
    "lub oil temp",       # useless
    "Coolant pressure"    # weak
])

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42,stratify=y
)
print("Columns in Xtrain:", Xtrain.columns.tolist())

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="Rajanan/ds-predictive-engine-maintenance-v1",
        repo_type="dataset",
    )
