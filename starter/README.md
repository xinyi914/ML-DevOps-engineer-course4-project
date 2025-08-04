It is a really simple case for praticing deploying a scalable ML pipeline
in production. The whole process includes data fetching, data cleaning, data training, inference, API creation and API deployment. 

# Environment Set up
source folder is ML-DevOps-engineer-course4-project/starter
run in python==3.10
in source folder run
```
pip install -r requirements.txt
```

AWS CLI to use the Access key ID and Secret Access key.

## Data

Used Adult dataset which is the census of the population for predicting whether a person makes over 50K a year. More details can be found in model_card

## Model

Used ramdomforest classifier. The parameters are in starter/param.yaml, the model is saved in starter/model.pkl. The functions of processing data and training model are in starter/ml/data.py and starter/ml/model.py.

## API Creation
 To run API,
 ```
 uvicorn main:app --reload
 ```
 and then open http://127.0.0.1:8000/docs


## API Deployment

Deployed in Render
run
```
python live_post.py 
```
to get sample request

