#
# Author: Jamey Johnston
# Title: Code Like a Snake Charmer: Advanced Data Modeling in Python!
# Date: 2019/11/08
# Blog: http://www.STATCowboy.com
# Twitter: @STATCowboy
# Git Repo: https://github.com/STATCowboy/SnakeCharmer-Adv
#

import requests, json

# url = 'http://127.0.0.1:5000/' # Local Flask Test
# url = 'http://127.0.0.1:8080/' # Docker Test
url = 'https://redwineapp.azurewebsites.net/' # Azure Web App Test

# "fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"

text = json.dumps({"0":{"fixed acidity":6.8, "volatile acidity":.47, "citric acid":.08, "residual sugar":2.2, "chlorides":.0064, "free sulfur dioxide":18.0, "total sulfur dioxide":38.0, "density":.999933, "pH":3.2, "sulphates":.64, "alcohol":9.8},
               	"1":{"fixed acidity":2.8, "volatile acidity":.43, "citric acid":.04, "residual sugar":3.2, "chlorides":.0054, "free sulfur dioxide":18.2, "total sulfur dioxide":38.2, "density":.999933, "pH":3.2, "sulphates":.64, "alcohol":6.8},
               	"2":{"fixed acidity":4.3, "volatile acidity":.44, "citric acid":.081, "residual sugar":2.3, "chlorides":.0074, "free sulfur dioxide":17.0, "total sulfur dioxide":37.1, "density":.999933, "pH":3.2, "sulphates":.64, "alcohol":10.8}})

headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}

r = requests.post(url, data=text, headers=headers)

print(r,r.text)
