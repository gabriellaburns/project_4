# from crypt import methods
from pydoc import allmethods
import numpy as np
import pandas as pd
import sqlalchemy
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, inspect, func

from flask import Flask, jsonify, render_template, request 

#################################################

#################################################
app = Flask(__name__)
#################################################
# Flask Routes
#################################################

# create route that renders index.html template
@app.route("/")
def index():

    return render_template("index.html") 

@app.route("/formurl", methods=["post"])
def get_values():

    # Get input values for the model from html
    input = []
    loan_amount = request.form["loanamount"] # for the input
    input.append(loan_amount)
    income = request.form["income"]
    input.append(income)
    loantype = request.form["type"]
    for i in loantype:
        input.append(i)
    aeth = request.form["aeth"]
    for i in aeth:
        input.append(i)
    coaeth = request.form["coaeth"]
    for i in coaeth:    
        input.append(i)
    arac = request.form["arac"]
    for i in arac:
        input.append(i)
    coarac = request.form["coarac"]
    for i in coarac:
        input.append(i)
    asex = request.form["asex"]
    for i in asex:
        input.append(i)
    coasex = request.form["coasex"]
    for i in coasex:
        input.append(i)

    print("---------------------------------------------")
    print(input)
    print("---------------------------------------------")

    import pickle


    model = pickle.load(open('resources/lr_classifier.pkl', 'rb'))       
    chance=0

    data = np.array(input)[np.newaxis, :]  # converts shape for model test
    predict = model.predict(data)  # runs model on the data
    prob= model.predict_proba(data)


    if predict == [1]:
        prediction = "We predict success, congratulations!"
    else:
        prediction = "Sorry, you will likely not qualify."

    prob= model.predict_proba(data)

    if predict == [1]:
        chance = prob[0][1]
    else:
        chance = prob[0][0]

    print(prediction)
    print("---------")
    print(chance)

            
    model_result = {
        "prediction":prediction,
        "probability":str((chance*100)) + '%'
    }
    # Load he model and feed the input to the model to get the result
    # model_result = michales_fantastic_model_result

    # Render the result through a html page
    return render_template("result.html", result = model_result) 
    #return render_template("index.html",)    



if __name__ == '__main__':
    app.run(debug=True)
