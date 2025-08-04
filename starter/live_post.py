import requests

live_URL = 'https://ml-devops-engineer-course4-project.onrender.com/predict'

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
response = requests.post(live_URL,json=data)

print("Status code: ", response.status_code)
print("Result: ", response.json())