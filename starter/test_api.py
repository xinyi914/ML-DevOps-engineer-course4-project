from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

def test_api_get():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"greeting": "Welcome"}

def test_post_one_sample():
    data = {
        "age": [30],
        "workclass":["Private"],
        "fnlgt": [39460],
        "education": ["Bachelors"],
        "education-num": [5],
        "marital-status": ["Married-civ-spouse"],
        "occupation": ["craft-repair"],
        "relationship": ["Husband"],
        "race": ["Black"],
        "sex": ["Male"],
        "capital-gain": [0],
        "capital-loss": [0],
        "hours-per-week": [40],
        "native-country": ["United-States"]
    }
    response = client.post("/predict",json=data)
    result = response.json()
    assert response.status_code == 200
    assert result["prediction"] in [0,1]

def test_post_lists():
    data = {
        "age": [30,40],
        "workclass": ["Private","self-emp-inc"],
        "fnlgt": [39460,39461],
        "education": ["Bachelors","Masters"],
        "education-num": [5,7],
        "marital-status": ["Married-civ-spouse","Divorced"],
        "occupation": ["craft-repair","Tech-support"],
        "relationship": ["Husband","Wife"],
        "race": ["Black","White"],
        "sex": ["Male","Female"],
        "capital-gain": [0,0],
        "capital-loss": [0,0],
        "hours-per-week": [40,40],
        "native-country": ["United-States","United-States"]
    }
    response = client.post("/predict",json=data)
    result = response.json()
    assert response.status_code == 200
    assert len(result["prediction"]) == 2

