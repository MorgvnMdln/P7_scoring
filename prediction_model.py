import joblib
import pandas as pd
import pickle
import string
import sklearn
import scipy
import lime
from pickle import load
import dill as pickle

# ----------------------------------------------------
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

# ----------------------------------------------------
from lightgbm import LGBMClassifier

# ----------------------------------------------------


print('Tags transformer loaded.')
from scipy import sparse
import re
import numpy as np

class PredictionModel:
    MODEL_PATH = './model/model_global.pkl'
    LIME_PATH = './model/lime_global.pkl'
    LOCAL_FEAT_IMPORTANCE_PATH = './model/feature_importance_locale.txt'
    TEST_DATA_PATH = './model/donnees_test.pkl'
    local_feat_importance = None
    test_data = None

    def __init__(self) -> None:
        self._model = self.import_predict_model()  # load lgbm
        self.explainer = self.import_lime_model()  # laod explainer
        self.test_data = pd.read_pickle(self.TEST_DATA_PATH)  # load dataframe donnees test
        # self.load_feat_importance_local()  # self.local_feat_importance = self.load_feat_importance_local()
        pass

    def load_features(self, client_Id):
        print('Dataframe :\n', self.test_data.head())
        # Drop SK_ID_CURR pour modeliser selon le nombre de features du notebook
        return self.test_data.loc[self.test_data['SK_ID_CURR'] == int(client_Id)]
        # return self.test_data.loc[self.test_data['SK_ID_CURR'] == int(client_Id)].to_numpy()
        # return self.test_data.loc[self.test_data['SK_ID_CURR'] == int(client_Id)].drop(columns=['SK_ID_CURR']).to_numpy()
        # return self.test_data.loc[self.test_data['SK_ID_CURR'] == float(client_Id)].to_numpy()


    def predict(self, client_Id):
        arr_results = []
        # features = self.load_features(np.delete(client_Id, -1, axis=1))  # donnees du client retournees sans SK_ID_CURR
        features = self.load_features(client_Id)  # donnees du client retournees
        features = features.loc[:, features.columns != 'SK_ID_CURR'].to_numpy() # !!! => ajout

        # preparation input
        print('Loaded features shape: ', features.shape)

        # prediction
        arr_results = self._model.predict_proba(features)

        # prediction Local Interpretable Model-agnostic Explanations

        # client_index = self.test_data[self.test_data['SK_ID_CURR'] == float(client_Id)].index
        # print(self.test_data.loc[self.test_data['SK_ID_CURR'] == int(client_Id)])
        print('Indexe du numero client :', np.where(self.test_data['SK_ID_CURR']==int(client_Id))[0][0])  # Afficher l'index de SK_ID_CURR
        client_index = np.where(self.test_data['SK_ID_CURR']==int(client_Id))[0][0]
        print('debug : ', self.test_data.iloc[client_index, 0:-1])


        # client_features = self.test_data.iloc[client_index]  # Recuperer les features du client
        # client_features = client_features.drop(columns=['SK_ID_CURR'])  # Drop SK_ID_CURR pour expliquer la participation des features
        # client_features = client_features.to_numpy()
        # df.drop('SK_ID_CURR', axis=1, inplace=True)
        # self.test_data.iloc[client_index].drop('SK_ID_CURR', axis=1, inplace=True)
        self.local_feat_importance = self.explainer.explain_instance(self.test_data.iloc[client_index, 0:-1],
                                                       self._model.predict_proba,
                                                       num_samples=100)  # passer X en format numpy array
        '''local_feat_importance = self.explainer.explain_instance(self.test_data.iloc[client_index],
                                                       self._model.predict_proba,
                                                       num_samples=100)  # passer X en format numpy array'''
        temp_ = self.local_feat_importance.as_list()
        print ('temp_ :\n', temp_)
        # print('local_feat_importance :\n{}'.format(local_feat_importance))

        # conversion vecteur en tags
        label = {'numéro client' : client_Id, 'label_1':'good', 'score_1':float(arr_results[0][0]),'label_2':'bad', 'score_2':float(arr_results[0][1]),\
                 'feature_importance_locale':dict(self.local_feat_importance.as_list())}
        print(f'label predicted :\n{label}')
        return label


    def model_performance(self, X):
        info_perf= ''
        return info_perf
    
    def import_predict_model(self):
        model = joblib.load(self.MODEL_PATH)
        return model

    def import_lime_model(self):
        with open(self.LIME_PATH, 'rb') as file:
            explainer = pickle.load(file)
        return explainer

if __name__ =='__main__':
    test = PredictionModel()
    input_test = {}
    input_test['txt_question'] = 'Convert a Python list of lists to a single string'
    input_test['txt_body'] = "<p>I have list of lists consisting of chars and integers that I want to convert into a single string"
    test.predict(input_test)