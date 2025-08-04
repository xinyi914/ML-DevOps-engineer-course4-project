import pytest
import numpy as np
import pandas as pd
import yaml
from yaml import CLoader as Loader

# Add the necessary imports for the starter code.
from sklearn.ensemble import RandomForestClassifier

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from ml.model import train_model, compute_model_metrics,inference

@pytest.fixture(scope="session")
def params():
    path = os.path.join(os.path.dirname(__file__), "param.yaml")
    with open(path) as f:
        params = yaml.load(f,Loader=Loader)
    return params

def test_train_model(params):
    X_train = np.array([[0,1],[1,0],[0,0]])
    y_train = np.array([0,1,0])
    model = train_model(X_train,y_train,params)
    assert isinstance(model,RandomForestClassifier)

def test_inference(params):
    X_train = np.array([[0,1],[1,0],[0,0]])
    y_train = np.array([0,1,0])
    X_test = np.array([[1,1]])
    model = train_model(X_train,y_train,params)
    preds = inference(model,X_test)
    assert isinstance(preds,np.ndarray)
    assert preds.shape == (1,)

   
def test_compute_model_metrics(params):
    X_train = np.array([[0,1],[1,0],[0,0]])
    y_train = np.array([0,1,0])
    X_test = np.array([[1,1]])
    y_test = np.array([1])
    model = train_model(X_train,y_train,params)
    preds = inference(model,X_test)
    precision, recall, fbeta = compute_model_metrics(y_test,preds)
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1