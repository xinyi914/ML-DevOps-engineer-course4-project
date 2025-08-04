# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Union, List

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from starter.ml.model import inference
from starter.ml.data import process_data
import pandas as pd
import joblib

app = FastAPI()

class InputData(BaseModel):
    age: List[int] = Field(default=None,examples=[30])
    workclass: List[str] = Field(default=None,examples=["Private"])
    fnlgt: List[int] = Field(default=None,examples=[13453])
    education: List[str] = Field(default=None,examples=["Bachelors"])
    education_num: List[int] = Field(default=None,alias="education-num",examples=[8])
    marital_status:List[str]= Field(default=None,alias="marital-status",examples = ["Divorced"])
    occupation: List[str] = Field(default=None,examples=["Tech-suuport"])
    relationship: List[str] = Field(default=None,examples=["Wife"])
    race: List[str] = Field(default=None,examples=["White"])
    sex: List[str] = Field(default=None,examples=["Female"])
    capital_gain: List[int] = Field(default=None,alias="capital-gain",examples = [0])
    capital_loss: List[int] = Field(default=None,alias="capital-loss",examples = [0])
    hours_per_week: List[int] = Field(default=None,alias="hours-per-week",examples = [40])
    native_country: List[str] = Field(default=None,alias="native-country",examples = ["United-States"])
    class Config:
        validate_by_name = True  # Allows you to use 'education_num' in code, 'education-num' in JSON
        allow_population_by_alias = True 
 

@app.get("/")
async def say_welcome():
    return {"greeting": "Welcome"}

@app.post("/predict")
async def predict(data: InputData):
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
    print("data transformation")
    input_dict = data.dict(by_alias=True)
    print("to dataframe")
    if isinstance(next(iter(input_dict.values())),list):
        df = pd.DataFrame(input_dict)
    else:
        df = pd.DataFrame([input_dict])
    print("download encoder, lb")
    path_encoder = os.path.join(os.path.dirname(__file__), "starter/encoder.joblib")
    path_lb = os.path.join(os.path.dirname(__file__), "starter/label_binarizer.joblib")
    path_model = os.path.join(os.path.dirname(__file__), "starter/model.pkl")
    encoder = joblib.load(path_encoder)
    lb = joblib.load(path_lb)
    model = joblib.load(path_model)
    print("process data")
    X,_,_,_ = process_data(
    df, categorical_features=cat_features, label=None, encoder = encoder, training=False, lb=lb)
    # print(X.shape)

    print("inference")
    preds = inference(model,X)
    if len(preds) == 1:
        return {"prediction": int(preds[0])}
    return {"prediction": preds.tolist()}