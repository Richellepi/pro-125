from re import DEBUG
from flaskm import Flask,jsonify,request
from classifier import get_prediction

app=Flask(__name__)

@app.route("/predict-alpbet",methods=[POST])
def predict_data():
    image=request.files.get("alphabet")
    predicition=get_prediction(image)
    return jsonify({
    "prediciton":predicition
}),200

if(__name__,"__main__"):
    app.run(debug=True)