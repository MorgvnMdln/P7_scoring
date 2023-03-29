import streamlit as st
import datetime
from datetime import date
import pandas as pd
import joblib
import requests
import json
import argparse
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
import csv
import io, base64
from PIL import Image


# ----------------------------------------------------
from sklearn.preprocessing import MinMaxScaler
# ----------------------------------------------------
from lightgbm import LGBMClassifier
# ----------------------------------------------------

def get_days(date_value):
    today = date.today()
    delta = today - date_value
    return delta.days



# make a title for your webapp
st.title("Formulaire Credit Scoring")

# Documentation >> st.help(st.form)

tab1, tab2 = st.tabs(["Client Information", "Global Performance"])
with tab1:  # ID client + resultats après réponse de l'API
    # Creating our form fields
    with st.form("Formulaire Credit Scoring", clear_on_submit=True):
        uff, col, buff2 = st.columns([1,3,1])
        # (SK_ID_CURR) Numéro client
        sk_id_curr = col.text_input('Entrez un numéro de client')

        submit = st.form_submit_button(label="Envoyer")

        # If submit button is pressed
        if submit:
            # app.py
            URL = "https://credit-scoring-app-mdln.herokuapp.com/api/predict"
            # URL = "http://127.0.0.1:5000/api/predict"

            # defining a params dict for the parameters to be sent to the API
            PARAMS = {
                  "sk_id_curr": sk_id_curr
                     }
            # Compute the client score
            # print(PARAMS)
            # sending get request and saving the response as response object
            r = requests.get(url=URL, params=PARAMS)
            # extracting data in json format
            response = r.json()
            label = response['data']
            #   print(response)
            print(label)
            label2 = (label['score_1'], label['score_2'])
            print(label2)

            # PieChart
            if submit:
                    st.header("Client Information")
                    # Customer score visualization
                    st.write("**Synthèse des informations du client n°{}**".format(sk_id_curr))
                    fig, ax = plt.subplots(figsize=(5,5))
                    plt.pie(label2, explode=[0, 0.1], labels=['Good', 'Bad'], autopct='%1.1f%%', startangle=90)
                    # Change the background color of the plot
                    # plt.rcParams['figure.facecolor'] = 'black'
                    fig.patch.set_facecolor('#262730')
                    # st.sidebar.pyplot(fig)
                    st.pyplot(fig)

                    # Local visualization
                    feature_importance_local = label['feature_importance_locale']
                    print(feature_importance_local)
                    feature_importance_local = pd.DataFrame(feature_importance_local.items(), columns=['Feature', 'Value'])
                    print('feature_importance_local :\n', feature_importance_local)

                    # st.sidebar.header("**Local features importances contribution**")
                    st.header("**Local features importances contribution**")
                    fig = plt.figure(figsize=(10, 8), facecolor='#262730')  # facecolor=('#262730')
                    plt.barh(feature_importance_local['Feature'], sorted(feature_importance_local['Value']))
                    plt.xlabel("Value")
                    plt.ylabel("Feature")
                    plt.tight_layout()
                    # fig.patch.set_facecolor('#262730')
                    fig.set_facecolor('#262730')
                    # st.sidebar.pyplot(fig)
                    st.pyplot(fig)
                    #fig, ax = plt.subplots(figsize=(10,5))
                    # st.bar_chart(feature_importance_globale, x=feature_importance_globale.columns())
                    # ax.hist(feature_importance_globale, bins=20)
                    # st.pyplot(fig)

    with tab2: #onglet performance du model
        URL = "http://credit-scoring-app-mdln.herokuapp.com/api/model_performance"
        # URL = "http://127.0.0.1:5000/api/model_performance"
        # defining a params dict for the parameters to be sent to the API

        st.header("Global Performance")
        r = requests.get(url=URL)
        # extracting data in json format
        print(r)
        response = r.json()
        #   print(response)

        # images-by-stream
        image0 = Image.open(io.BytesIO(base64.decodebytes(bytes(response['features_importances'], "utf-8"))))
        st.image(image0, caption='Feature Importance Global')

        image1 = Image.open(io.BytesIO(base64.decodebytes(bytes(response['confusion_matrix'], "utf-8"))))
        st.image(image1, caption='Confusion Matrix')

        image2 = Image.open(io.BytesIO(base64.decodebytes(bytes(response['roc_auc'], "utf-8"))))
        st.image(image2, caption='Area Under The Curve')

