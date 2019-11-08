from flask import Flask

#
# https://docs.microsoft.com/en-us/azure/python/tutorial-deploy-app-service-on-linux-01
#

myapp = Flask(__name__)

@myapp.route("/")
def hello():
    return "Hello Pass Summit 2019 Attendees!"