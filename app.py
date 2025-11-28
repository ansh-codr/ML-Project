from flask import Flask, request, jsonify, render_template
import os
from model import predict_patient, ensure_ready

app = Flask(__name__, template_folder="templates", static_folder="static")

ensure_ready()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True) or {}
    prob = predict_patient(data)
    rec = "recommend immunotherapy" if prob >= 0.5 else "do not recommend immunotherapy"
    return jsonify({"probability": prob, "recommendation": rec})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
