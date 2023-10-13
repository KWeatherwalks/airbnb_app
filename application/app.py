# Import Statements
import pandas as pd
from flask import Flask, render_template, request
from joblib import load

from .predict import get_prediction

# Instantiate Application
app = Flask(__name__)
load_model = load('finalized_model.sav')

# as easy as changing path to /form and make a link to it in main page
@app.route('/')
def form():
    return render_template('form.html')

@app.route('/run_model', methods=['POST', 'GET'])
def data():
    # if user types in /run_model they get this error message
    if request.method == 'GET':
        message = f"ERROR: The URL /run_model is accessed directly. Try going to home page '/' to submit form"
    # if user goes to /form and hits submit, they go to this page!
    # in here user input gets stored into to_predict, and then to_predict gets run in model
    if request.method == 'POST':
        property_type = str(request.values["prop"])
        room_type = str(request.values["room_type"])
        bathrooms = float(request.values["bathrooms"])
        cancellation_policy = str(request.values["cancellation"])
        city = str(request.values["city"])
        host_since = str(request.values["host_since"])
        review_scores_rating = int(request.values["review_rating"])
        bedrooms = int(request.values["bedrooms"])
        beds = int(request.values["beds"])
        # We will be adding a few more dropdowns above
        # need to get this to print out T/F list
        amenities = request.form.getlist('feature_checkbox')
        # basics =
        to_predict = [property_type, room_type, bathrooms,
                        cancellation_policy, city, host_since,
                        review_scores_rating, bedrooms, beds,
                        amenities]

        message = model_output(to_predict)
    return message

def model_output(user_input):

    mod_input = []

    all_amenities = [
        "instant_bookable",
        "host_has_profile_pic",
        "host_identity_verified",
        "cleaning_fee",
        "Wireless Internet",
        "Air conditioning",
        "Kitchen",
        "Heating",
        "Family/kid friendly",
        "Hair dryer",
        "Iron",
        "Shampoo",
        "Fire extinguisher",
        "Laptop friendly workspace",
        "Indoor fireplace",
        "TV",
        "Cable TV"]

    # Append unchanging variables to list first : check indexing there?
    mod_input.extend(user_input[:9])
    input = user_input[9]
    # For loop through conditional varibles
    for option in all_amenities:
        if any(option in s for s in input):
            mod_input.append(1)
        else:
            mod_input.append(0)
    price='${}'.format(get_prediction(mod_input, load_model))
    return render_template('results.html', prediction=price)
