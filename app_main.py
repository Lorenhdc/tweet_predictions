from flask import Flask, jsonify, request
import os
import pickle
from sklearn.linear_model import Lasso
import pandas as pd
from sklearn.model_selection import train_test_split

os.chdir(os.path.dirname(__file__))


app = Flask(__name__)
app.config['DEBUG'] = True

model = pickle.load(open('finished_model.model','rb'))

@app.route('/', methods=['GET'])
def home():
    return "<h1>Welcome to the tweet prediction portal</h1><p>This site is a prototype API for tweet sentiment predictions.</p>"


@app.route('/api/v1/predict', methods=['GET'])
def predict():

    tweet_serie = request.args.get('tweet', None)
    if tweet_serie != None:
        frase_predecir = "'"+tweet_serie+"'"

    if tweet_serie is None:
        return "Args empty, the data are not enough to predict"
    else:
        prediction = model.predict([frase_predecir])

        return str(prediction)



#app.run()