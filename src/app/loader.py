import joblib
import os
from flask import Flask, request, jsonify

MODEL_FILES = {
    'logistic': 'logistic_regression.pkl',
    'svm': 'svm.pkl',
    'tree': 'decision_tree.pkl',
    'knn': 'knn.pkl'
}

def create_app(models_path: str = "src/models"):
    app = Flask(__name__)
    scaler = joblib.load(os.path.join(models_path, "scaler.pkl"))
    models = {}
    for key, fname in MODEL_FILES.items():
        models[key] = joblib.load(os.path.join(models_path, fname))

    @app.route("/predict/<model_name>", methods=["POST"])
    def predict(model_name):
        if model_name not in models:
            return jsonify(error="Model no trobat"), 404
        data = request.json
        vals = data.get("features")
        if not isinstance(vals, list) or len(vals) != 8:
            return jsonify(error="Cal una llista de 8 valors"), 400
        import numpy as np
        X = np.array(vals).reshape(1, -1)
        Xs = scaler.transform(X)
        model = models[model_name]
        y_pred = int(model.predict(Xs)[0])
        proba = float(model.predict_proba(Xs)[0][y_pred])
        return jsonify(model=model_name, prediction=y_pred, probability=proba)

    return app
