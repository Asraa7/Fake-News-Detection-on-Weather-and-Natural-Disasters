import os

from flask import Flask, jsonify, render_template, request

from model import get_model_stats, predict_news, train_model

app = Flask(__name__)

if not os.path.exists("model.pkl") or not os.path.exists("stats.pkl"):
    train_model()


@app.route("/")
def index():
    return render_template("index.html", stats=get_model_stats())


@app.route("/compare")
def compare():
    return render_template("compare.html", stats=get_model_stats())


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    text = str(data.get("text", "")).strip()
    if len(text) < 10:
        return jsonify({"error": "Text too short. Please enter a proper news headline."}), 400
    if len(text) > 5000:
        return jsonify({"error": "Text too long (max 5000 characters)."}), 400
    return jsonify(predict_news(text))


@app.route("/stats")
def stats():
    return jsonify(get_model_stats() or {})


if __name__ == "__main__":
    app.run(debug=True)
