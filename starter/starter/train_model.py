# Script to train machine learning model.
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import yaml
from yaml import CLoader as Loader
import joblib
# Add the necessary imports for the starter code.
from ml.data import process_data
from ml.model import train_model,inference,compute_model_metrics,slice_performance

with open("./param.yaml") as f:
    params = yaml.load(f,Loader=Loader)

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


# logger.info("Load data")
# df = pd.read_csv("../data/census.csv")
# logger.info("Clean data")
# df_clean = df.map(lambda x: x.replace(" ","") if isinstance(x,str) else x)

# # save clean data
# logger.info("Save clean data")
# df_clean.to_csv("../data/census_clean.csv")
# Add code to load in the data.
logger.info("Load clean data")
data = pd.read_csv("../data/census_clean.csv")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
logger.info("Split clean data")
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Proces the test data with the process_data function.
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Train and save a model.
logger.info("Train and saving model")
clf = train_model(X_train,y_train,params)
joblib.dump(clf,"model.pkl")

# Inference
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", encoder = encoder, training=False, lb=lb
)

preds = inference(clf,X_test)
precision, recall, fbeta = compute_model_metrics(y_test,preds)
print("test results: ")
print(f"precision: {precision}, recall: {recall}, fbeta: {fbeta}")

for cat_fea in cat_features:
    print(f"feature: {cat_fea}")
    slice_performance(test,y_test,preds,cat_fea)

