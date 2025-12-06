import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for random numbers
from scipy.stats import randint, uniform
import random
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score
# for model serialization
import joblib
import mlflow
from sklearn.model_selection import RandomizedSearchCV
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("MLOps_experiment")

api = HfApi()

# Define constants for the dataset and output paths
Xtrain_path = "hf://datasets/asvravi/asv-tourism-package/Xtrain.csv"
Xtest_path = "hf://datasets/asvravi/asv-tourism-package/Xtest.csv"
ytrain_path = "hf://datasets/asvravi/asv-tourism-package/ytrain.csv"
ytest_path = "hf://datasets/asvravi/asv-tourism-package/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)

# List of numerical features in the dataset
numeric_features = [
    'Age',       # Customer's credit score
    'DurationOfPitch',
    'NumberOfFollowups',
    'PitchSatisfactionScore',
    'NumberOfPersonVisiting',
    'PreferredPropertyStar',
    'NumberOfTrips',
    'Passport',
    'OwnCar',
    'NumberOfChildrenVisiting',
    'MonthlyIncome',
    'CityTier',              # City category based on development, population, and living standards
]

# List of categorical features in the dataset
categorical_features = [
    'TypeofContact',         # Country where the customer resides
    'Occupation',            # Customer's occupation
    'Gender',                # Gender of the customer
    'ProductPitched',
    'MaritalStatus',         # Marital status of the customer
    'Designation',           # Designation of the customer's current position
]

# Set the clas weight to handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]
class_weight

# Define the preprocessing steps
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# Define base XGBoost model
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

# Define hyperparameter grid
param_grid = {
    'xgbclassifier__n_estimators': randint(50, 100),
    'xgbclassifier__max_depth': randint(2, 5),
    'xgbclassifier__colsample_bytree': uniform(0.4, 0.2),
    'xgbclassifier__colsample_bylevel': uniform(0.4, 0.2),
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
    'xgbclassifier__reg_lambda': uniform(0.4, 0.2),
}

# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Start MLflow run
with mlflow.start_run():
    # Hyperparameter tuning
    rand_search = RandomizedSearchCV(model_pipeline, param_grid, n_iter=30,cv=5, n_jobs=-1)
    rand_search.fit(Xtrain, ytrain)

    """
    # Log all parameter combinations and their mean test scores
    results = rand_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        std_score = results['std_test_score'][i]

        # Log each combination as a separate MLflow run
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", std_score)
    """
    
    # Log best parameters separately in main run
    mlflow.log_params(rand_search.best_params_)

    # Store and evaluate the best model
    best_model = rand_search.best_estimator_

    classification_threshold = 0.45

    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Log the metrics for the best model
    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1-score": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score']
    })

    # Save the model locally
    model_path = "best_toursim_package_model_v1.joblib"
    joblib.dump(best_model, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face
    repo_id = "asvravi/asv-tourism-package"
    repo_type = "model"

    # Step 1: Check if the space exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Space '{repo_id}' created.")

    # create_repo("asv-toursim-package-model", repo_type="model", private=False)
    api.upload_file(
        path_or_fileobj="best_toursim_package_model_v1.joblib",
        path_in_repo="best_toursim_package_model_v1.joblib",
        repo_id=repo_id,
        repo_type=repo_type
    )
