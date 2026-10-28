from flask import Flask, render_template, request
import numpy as np
import pickle,joblib

app = Flask(__name__)

# Load model
model =joblib.load(open("model.pkl", "rb"))

# Load scaler
scaler = joblib.load(open("scaler.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")
@app.route('/form')
def form():
    return render_template('form.html')



@app.route("/predict", methods=["POST"])
def predict():



    features = [

    float(request.form["Administrative"]),
    float(request.form["Administrative_Duration"]),
    float(request.form["Informational"]),
    float(request.form["Informational_Duration"]),
    float(request.form["ProductRelated"]),
    float(request.form["ProductRelated_Duration"]),
    float(request.form["BounceRates"]),
    float(request.form["ExitRates"]),
    float(request.form["PageValues"]),
    float(request.form["SpecialDay"]),
    float(request.form["Month"]),
    float(request.form["OperatingSystems"]),
    float(request.form["Browser"]),
    float(request.form["Region"]),
    float(request.form["TrafficType"]),
    float(request.form["VisitorType"]),
    float(request.form["Weekend"])

    ]
    
   

    features = np.array(features).reshape(1, -1)

    features = scaler.transform(features)

    prediction = model.predict(features)

    if prediction[0] == 1:
        result = "Customer Will Purchase"
    else:
        result = "Customer Will NOT Purchase"

    return render_template("result.html", prediction=result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)