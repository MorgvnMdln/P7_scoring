
from flask import Flask, render_template, redirect, url_for, request, jsonify
from werkzeug.wrappers import Request, Response
from .prediction_model import PredictionModel
import base64

pred_model = PredictionModel()
app = Flask(__name__)

@app.route("/api/predict", methods=["GET"])

def predict():
    args = request.args
    client_Id = args['sk_id_curr']
    print(client_Id)
    arr_results = None

    arr_results = pred_model.predict(client_Id)
    print(arr_results)
    return jsonify({
                    'status': 'ok',
                    'data': arr_results
                    })

@app.route("/api/model_performance", methods=["GET"])

def get_model_performance():
    return jsonify({
                    'status': 'ok',
                    'features_importances' :  imageToString('images/Feature_Importance_Globale.png'),  # localhost:5000/images/Classement des features les plus importantes.png
                    'confusion_matrix_auc': imageToString('images/Confusion_Matrix_AUC.png'),  # localhost:5000/images/Confusion_Matrix.png
                    'confusion_matrix': imageToString('images/Confusion_Matrix.png'),
                    'roc_auc' :  imageToString('images/AUC.png') # localhost:5000/images/ROC_Curve_Analysis.png 'A SUPPRIMER'
                    })

def imageToString(image_path):
    b64_string = ''
    with open(image_path, "rb") as img_file:
            b64_string = base64.b64encode(img_file.read())
    return b64_string.decode('utf-8')


if __name__ == "__main__":
    app.run(port=5000, debug=True)
