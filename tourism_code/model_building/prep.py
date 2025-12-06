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
DATASET_PATH = "hf://datasets/asvravi/asv-tourism-package/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

#Do the required data cleansing, imputations, transformations, feature engineering etc

#Fixing data entry error in Gender
df["Gender"] = df["Gender"].replace("Fe Male", "Female")

#Fixing the data types
cols = df.select_dtypes("object") 
# converting object dataype to category
for i in cols.columns:
    df[i] = df[i].astype("category")

#Dropping the columns that are not required for model
# Since the objective is to build models on data of the existing customers which can be used to target new customers, check if we can drop the customer interaction data also from the dataset as those features will not be available for new customers.
#Also, CustomerID will not be of much help in model building and hence dropping that too.

"""
df.drop(
    [
        "CustomerID",
        "DurationOfPitch",
        "NumberOfFollowups",
        "ProductPitched",
        "PitchSatisfactionScore",
    ],
    axis=1,
    inplace=True,
)
"""
df.drop(
    [
        "Unnamed: 0",
        "CustomerID"
    ],
    axis=1,
    inplace=True,
)

target_col = 'ProdTaken'

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

#print shapes of all above 4 datasets with labels
print("Xtrain shape:", Xtrain.shape)
print("Xtest shape:", Xtest.shape)
print("ytrain shape:", ytrain.shape)
print("ytest shape:", ytest.shape)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="asvravi/asv-tourism-package",
        repo_type="dataset",
    )
