#
# Author: Jamey Johnston
# Title: Code Like a Snake Charmer: Advanced Data Modeling in Python!
# Date: 2019/11/08
# Blog: http://www.STATCowboy.com
# Twitter: @STATCowboy
# Git Repo: https://github.com/STATCowboy/SnakeCharmer-Adv
#

from flask import Flask, request, redirect, url_for, flash, jsonify
import json, pickle
import pandas as pd
import numpy as np
import os
from settings import APP_STATIC

# To test locally in PS conda shell
# 
# 1.) $env:FLASK_APP="RedWineQualityFlaskRun:app"
# 2.) flask run

app = Flask(__name__)
@app.route('/', methods=['POST'])

def makecalc():
    """
    Function run at each API call
    """
    modelfile = 'winequality-red.pickleRF.dat'
    model = pickle.load(open(os.path.join(APP_STATIC, modelfile), 'rb'))

    jsonfile = request.get_json()
    data = pd.read_json(json.dumps(jsonfile),orient='index')
    print(data)
    
    res = dict()

    # Headers of Data
    # "fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"

    # create array from JSON data from service
    X = np.array(data[['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']])

    print(X)

    ypred = model.predict(X)
    
    for i in range(len(ypred)):
        res[i] = ypred[i]
    
    return jsonify(res) 

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8080)
