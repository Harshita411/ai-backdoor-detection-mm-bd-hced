from flask import Flask, request, jsonify
from flask_cors import CORS

from utils.model_loader import load_model
from utils.data_loader import get_dataloader

from detectors.mmbd import run_mmbd
from detectors.strip import run_strip
from detectors.hybrid import run_hybrid

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return {"message": "Backend running"}

@app.route("/api/detect", methods=["POST"])
def detect():
    data = request.get_json()

    model_path = data.get("model_path")

    model = load_model(model_path)
    dataloader = get_dataloader()

    mmbd = run_mmbd(model)
    strip = run_strip(model, dataloader)
    hybrid = run_hybrid(model, dataloader, mmbd)

    return jsonify({
        "mmbd": mmbd,
        "strip": strip,
        "hybrid": hybrid
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)