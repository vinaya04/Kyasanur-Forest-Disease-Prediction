# import warnings
# warnings.filterwarnings("ignore", category=UserWarning)
#
# import json
# import joblib
# import numpy as np
# from flask import Flask, request, jsonify, render_template
# from flask_cors import CORS
#
# # ─── Flask Setup ────────────────────────────────────────────────────────────────
# app = Flask(__name__, static_folder='static', template_folder='templates')
# CORS(app)
#
# # ─── Load Artifacts ─────────────────────────────────────────────────────────────
# model = joblib.load("best_kfd_model.pkl")
# scaler = joblib.load("scaler.pkl")
# selected_features = joblib.load("selected_features.pkl")
#
# with open("metrics.json", "r") as f:
#     metrics = json.load(f)
#
# class_mapping = {
#     0: 'Confirmed (C)',
#     1: 'Suspected (S)',
#     2: 'Negative (N)',
#     3: 'Probable (PR)'
# }
#
# # ─── Page Routes ────────────────────────────────────────────────────────────────
#
# @app.route('/')
# def home():
#     return render_template("index.html")
#
# @app.route('/about')
# def about():
#     return render_template("about.html")
#
# @app.route('/precautions')
# def precautions():
#     return render_template("precautions.html")
#
# # ─── Prediction API ─────────────────────────────────────────────────────────────
#
# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         data = request.get_json()
#
#         if not data or "features" not in data:
#             return jsonify({"error": "No features provided"}), 400
#
#         features = data["features"]
#
#         if len(features) != len(selected_features):
#             return jsonify({"error": "Invalid number of features"}), 400
#
#         # Scale and predict
#         scaled = scaler.transform([features])
#         pred_index = int(model.predict(scaled)[0])
#         label = class_mapping[pred_index]
#
#         return jsonify({
#             "prediction": pred_index,
#             "class_label": label,
#             "model_accuracy": metrics.get("accuracy", "N/A")
#         })
#
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
#
# # ─── Metrics API ────────────────────────────────────────────────────────────────
#
# with open("metrics.json", "r") as f:
#     metrics = json.load(f)
#
# # ─── Run App ────────────────────────────────────────────────────────────────────
#
# if __name__ == '__main__':
#     app.run(debug=True)




#
# import warnings
# import json
# import joblib
# import numpy as np
# from flask import Flask, request, jsonify, render_template
# from flask_cors import CORS
#
# # ─── Flask Setup ────────────────────────────────────────────────────────────────
# app = Flask(__name__, static_folder='static', template_folder='templates')
# CORS(app)
#
# # ─── Load Artifacts ─────────────────────────────────────────────────────────────
# model = joblib.load("best_kfd_model.pkl")  # Load your trained model
# scaler = joblib.load("scaler.pkl")  # Load scaler for feature scaling
# selected_features = joblib.load("selected_features.pkl")  # Load selected features for prediction
#
# with open("metrics.json", "r") as f:
#     metrics = json.load(f)  # Load model metrics
#
# class_mapping = {
#     0: 'Confirmed (C)',
#     1: 'Suspected (S)',
#     2: 'Negative (N)',
#     3: 'Probable (PR)'
# }
#
# # ─── Page Routes ────────────────────────────────────────────────────────────────
#
# @app.route('/')
# def home():
#     return render_template("index.html")
#
# @app.route('/about')
# def about():
#     return render_template("about.html")
#
# @app.route('/precautions')
# def precautions():
#     return render_template("precautions.html")  # Corrected "precautions" page path
#
#
# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         if request.is_json:
#             data = request.get_json()
#             features = data.get("features", [])
#         else:
#             # Form data
#             features = [int(request.form.get(f"feature{i}", 0)) for i in range(len(selected_features))]
#
#         if len(features) != len(selected_features):
#             return jsonify({"error": "Invalid number of features"}), 400
#
#         scaled = scaler.transform([features])
#         pred_index = int(model.predict(scaled)[0])
#         label = class_mapping.get(pred_index, "Unknown")
#
#         if request.is_json:
#             return jsonify({
#                 "prediction": pred_index,
#                 "class_label": label,
#                 "model_accuracy": metrics.get("accuracy", "N/A")
#             })
#         else:
#             # If form, return result via Jinja (optional), or JSON
#             return jsonify({
#                 "label": label,
#                 "accuracy": metrics.get("accuracy", "N/A")
#             })
#
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
#
# # ─── Run App ────────────────────────────────────────────────────────────────────
#
# if __name__ == '__main__':
#     app.run(debug=True)


#
# import warnings
# import json
# import joblib
# import numpy as np
# from flask import Flask, request, jsonify, render_template
# from flask_cors import CORS
#
# app = Flask(__name__, static_folder='static', template_folder='templates')
# CORS(app)
#
# # Load Artifacts
# model = joblib.load("best_kfd_model.pkl")
# scaler = joblib.load("scaler.pkl")
# selected_features = joblib.load("selected_features.pkl")
#
# with open("metrics.json", "r") as f:
#     metrics = json.load(f)
#
# class_mapping = {
#     0: 'Confirmed (C)',
#     1: 'Suspected (S)',
#     2: 'Negative (N)',
#     3: 'Probable (PR)'
# }
#
# def predict_disease(input_list):
#     # Wrap input in a DataFrame with correct feature names
#     input_df = pd.DataFrame([input_list], columns=selected_features)
#
#     # Scale the input
#     input_scaled = scaler.transform(input_df)
#
#     # Predict
#     prediction = best_model.predict(input_scaled)
#     return prediction[0]
#
# @app.route('/')
# def home():
#     return render_template("index.html")
# @app.route('/about')
# def about():
#     return render_template("about.html")
#
# @app.route('/precautions')
# def precautions():
#     return render_template("precautions.html")  # Corrected "precautions" page path
#
#
# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         features = [int(request.form.get(f"feature{i}", 0)) for i in range(len(selected_features))]
#
#         if len(features) != len(selected_features):
#             return jsonify({"error": "Invalid number of features"}), 400
#
#         scaled = scaler.transform([features])
#         pred_index = int(model.predict(scaled)[0])
#         label = class_mapping.get(pred_index, "Unknown")
#
#         return jsonify({
#             "label": label,
#             "accuracy": metrics.get("accuracy", "N/A")
#         })
#
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
#
# if __name__ == '__main__':
#     app.run(debug=True)


import warnings
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Load Artifacts
model = joblib.load("Models/best_kfd_model.pkl")
scaler = joblib.load("Models/scaler.pkl")
selected_features = joblib.load("Models/selected_features.pkl")  # A list of feature names

with open("dataset/metrics.json", "r") as f:
    metrics = json.load(f)

class_mapping = {
    0: 'Confirmed (C)',
    1: 'Suspected (S)',
    2: 'Negative (N)',
    3: 'Probable (PR)'
}

def predict_disease(input_dict):
    input_df = pd.DataFrame([input_dict])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    return prediction[0]

@app.route('/')
def home():
    return render_template("index.html", selected_features=selected_features)

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/precautions')
def precautions():
    return render_template("precautions.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Build a dict with correct feature names
        input_data = {}
        for feature in selected_features:
            value = int(request.form.get(feature, 0))
            input_data[feature] = value

        # Predict
        predicted_class = predict_disease(input_data)
        label = class_mapping.get(predicted_class, "Unknown")

        return jsonify({
            "label": label,
            "accuracy": metrics.get("accuracy", "N/A")
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
