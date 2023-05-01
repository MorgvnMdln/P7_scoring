
from flask import Flask, render_template, redirect, url_for, request, jsonify
from werkzeug.wrappers import Request, Response
# from prediction_model import PredictionModel
import base64
import pandas as pd
from pickle import load
import pickle
import numpy as np


# pred_model = PredictionModel()
app = Flask(__name__)
MODEL_PATH = 'model_global.pkl'
LIME_PATH = 'lime_global.pkl'
LOCAL_FEAT_IMPORTANCE_PATH = 'feature_importance_locale.txt'
TEST_DATA_PATH = 'donnees_test.json'

def load_pickle(path): 
    result = None
    pickle_in = open(path, "rb")
    result = load(pickle_in)
    pickle_in.close()
    return result 


@app.route("/api/predict", methods=["GET"])


def predict():
    args = request.args
    client_Id = args['sk_id_curr']
    print(client_Id)
    arr_results = []
    test_data = pd.read_json(TEST_DATA_PATH)
    features = test_data.loc[test_data['SK_ID_CURR'] == int(client_Id)]  # donnees du client retournees
    features = features.loc[:, features.columns != 'SK_ID_CURR'].to_numpy() # !!! => ajout
    _model = load_pickle(MODEL_PATH)
    explainer = load_pickle(LIME_PATH)

    # preparation input
    print('Loaded features shape: ', features.shape)

    # prediction
    arr_results = _model.predict_proba(features)

    # prediction Local Interpretable Model-agnostic Explanations
    print('Indexe du numero client :', np.where(test_data['SK_ID_CURR']==int(client_Id))[0][0])  # Afficher l'index de SK_ID_CURR
    client_index = np.where(test_data['SK_ID_CURR']==int(client_Id))[0][0]
    print('debug : ', test_data.iloc[client_index, 0:-1])

    local_feat_importance = explainer.explain_instance(test_data.iloc[client_index, 0:-1],
                                                    _model.predict_proba,
                                                    num_samples=100)  # passer X en format numpy array
    '''local_feat_importance = self.explainer.explain_instance(self.test_data.iloc[client_index],
                                                    self._model.predict_proba,
                                                    num_samples=100)  # passer X en format numpy array'''
    temp_ = local_feat_importance.as_list()
    print ('temp_ :\n', temp_)

    # conversion vecteur en tags
    label = {'numero client':client_Id, 'label_1':'good', 'score_1':float(arr_results[0][0]),'label_2':'bad', 'score_2':float(arr_results[0][1]),\
                'feature_importance_locale':dict(local_feat_importance.as_list())}
    print(f'label predicted :\n{label}')
    print(label)
    return jsonify({
                    'status': 'ok',
                    'data': label
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
