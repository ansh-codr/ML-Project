from flask import Flask, request, jsonify, render_template
import os
from model import predict_patient, load_model, preprocess, train

app = Flask(__name__)

@app.before_first_request
def warm_up():
    if not os.path.exists("data/processed/processed.csv"):
        preprocess()
    if not os.path.exists("model.pkl"):
        train()
    load_model()

@app.get("/")
def index():
    return render_template("index.html")

@app.post("/predict")
def predict():
    data = request.get_json(force=True) or {}
    prob = predict_patient(data)
    rec = "recommend immunotherapy" if prob >= 0.5 else "do not recommend immunotherapy"
    return jsonify({"probability": prob, "recommendation": rec})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
