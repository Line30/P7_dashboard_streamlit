from ctypes import string_at
from matplotlib.figure import Figure
import streamlit as st
import pandas as pd
import numpy as np
import requests
from PIL import Image
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import json
import seaborn as sns
import joblib
import time
import shap
import os
import plotly.graph_objects as go
import pickle
from traitlets.traitlets import default

# API Flask :
#URL_API = ' http://127.0.0.1:5000/'

header = st.container()
dataset = st.container()
features = st.container()
model_trainig = st.container()
model_explainer = st.container()
similar_clients = st.container()

# Load the joblib model LR
model_LR = joblib.load(open("models//LR_pipeline.joblib", "rb"))
# Load the joblib model KNN
model_knn = joblib.load('models//pipeline_KNN.joblib')

# test and train data
data_test_app = pd.read_csv('data_API/X_test_app_sample.csv.zip')
data_test_app = data_test_app.drop(['Unnamed: 0'], axis= 1)
data_client_infos = pd.read_csv('data_API/data_client_infos.csv.zip')
data_client_infos = data_client_infos.drop(['Unnamed: 0'], axis= 1)
data_exploration = pd.read_csv('data_API/data_exploration.csv.zip')
data_exploration = data_exploration.drop(['Unnamed: 0'], axis= 1)
data_train_app = pd.read_csv('data_API/x_train_app.csv.zip')
data_train_app = data_train_app.drop(['Unnamed: 0'], axis= 1)

# description of features
description = pd.read_csv('data_API/description.csv', encoding= 'unicode_escape')
description = description.drop(['Unnamed: 0', 'Special'], axis=1)

feats = [f for f in data_test_app.columns if f not in ['SK_ID_CURR', 'TARGET']]
x_test = data_test_app[feats]
x_train = data_train_app[feats]

# list of id clients
id_clients = data_test_app["SK_ID_CURR"].sort_values()
id_clients = id_clients.values
id_clients = pd.DataFrame(id_clients)

# list of features
ls_features = description["Row"]
ls_features = ls_features.values
ls_features = pd.DataFrame(ls_features)

with header:
    st.title('Welcome to this Dashboard')
    st.write('Our financial company **"Prêt à dépenser"** wants to implement a credit scoring tool to calculate the probability that a customer will repay his credit, and then classify the request as granted or denied credit.')

logo = Image.open("data_API/logo.png")
st.sidebar.image(logo,width=300)

with dataset:
    st.header('Data exploration : ')
    st.write('Credit dataset available on kaggle : https://www.kaggle.com/c/home-credit-default-risk/data')
    st.write('Credit refused, customer with bankruptcy risk : **Target = 1**')
    st.write('Credit granted, non-bankrupt customer : **Target = 0**')
    # Set the style of plots
    #plt.style.use('seaborn')
    #plt.style.use('dark_background')

    # preparing data to display in a bar chart
    distribution_income_type_gender = data_exploration[['CODE_GENDER', 'NAME_INCOME_TYPE', 'SK_ID_CURR']].groupby(
        ['CODE_GENDER', 'NAME_INCOME_TYPE']).agg('count').reset_index()
    distribution_education_gender = data_exploration[['CODE_GENDER', 'NAME_EDUCATION_TYPE', 'SK_ID_CURR']].groupby(
        ['CODE_GENDER', 'NAME_EDUCATION_TYPE']).agg('count').reset_index()

    #fig = plt.figure(figsize=(8, 6))
    sns.barplot(x="NAME_EDUCATION_TYPE", y="SK_ID_CURR", hue="CODE_GENDER", data=distribution_education_gender)
    plt.legend(prop={'size': 12}, bbox_to_anchor=(1, 0), loc='lower left')
    plt.title('Type of education of clients by gender', size=18)
    plt.xlabel('Education type', size=12)
    plt.ylabel('frequency', size=12)
    plt.xticks(rotation=90)
    plt.xticks(size=12)
    plt.yticks(size=12)
    #st.pyplot(fig)

    #fig = plt.figure(figsize=(8, 6))
    sns.histplot(data_exploration, x="AMT_INCOME_TOTAL", hue="CODE_GENDER", bins=100, multiple="stack")
    plt.legend(prop={'size': 12}, bbox_to_anchor=(1, 0), loc='lower left')
    plt.title('Total income amount of clients by gender', size=18)
    plt.xlabel('Total income (euro)', size=12)
    plt.ylabel('frequency', size=12)
    plt.xticks(rotation=0)
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.xlim([0, 8e5])
    #st.pyplot(fig)

    # Affichage d'informations de la base de données des clients dans la sidebar
    st.sidebar.header(":bar_chart:Exploration of the data set : ")
    visualization_type = ["Data description", "Univariate analysis", "Bivariate analysis"]
    visualizations = st.sidebar.multiselect("Visualization of the data set :", visualization_type)

    for visualization in visualizations:
        if visualization == "Data description":
            st.markdown(" #### Data description :")
            st.write(data_exploration.head(3))

            desc_feat = st.selectbox("Select feature", ls_features)
            if st.button('submit'):
                desc = description[description.Row==desc_feat].reset_index()
                st.write(desc["Description"][0])
            if st.checkbox('Show the description of the different tables'):
                img_description = Image.open("data_API/home_credit.png")
                st.image(img_description, width=800)
        elif visualization == "Univariate analysis":
            # Exploration univariée des variables
            st.markdown('#### Univariate analysis of features :')
            variables = ['Age', 'Gender', 'Family status', "Education type", "Working years", "Income type",
                         "Total income amount", "Contract type", "Credit amount", "Annuity amount"]
            features = st.multiselect("The features to be displayed:", variables)
            for feature in features:
                # Set the style of plots
                plt.style.use('seaborn')
                fig = plt.figure(figsize=(8, 6))
                if feature == 'Age':
                    # Plot the distribution of feature
                    st.write('<u>Age of Clients</u>', unsafe_allow_html=True)
                    sns.histplot(data_exploration["DAYS_BIRTH"].values / -365, edgecolor='k', bins=35, color='steelblue')
                    plt.title('Distribution of clients by age ', size=9)
                    plt.xlabel('Age (years)', size=9)
                    plt.ylabel('Frequency', size=9)
                    plt.xticks(size=9)
                    plt.yticks(size=9)
                    st.pyplot(fig)

                elif feature == 'Working years':
                    st.write('<u>Working years</u>', unsafe_allow_html=True)
                    sns.histplot(data_exploration['DAYS_EMPLOYED'].values / -365, edgecolor='k', color='steelblue')
                    plt.title('Distribution of clients by number of Working years', size=9)
                    plt.xlabel('Years of employment (Years)', size=9)
                    plt.ylabel('Frequency', size=9)
                    plt.xticks(size=9)
                    plt.yticks(size=9)
                    plt.xlim([0, 40])
                    st.pyplot(fig)

                elif feature == 'Gender':
                    st.write('<u>Gender of client </u>', unsafe_allow_html=True)
                    # preparing data to display on pie chart
                    client_gender = data_exploration['CODE_GENDER'].value_counts().reset_index()
                    client_gender.columns = ['Gender', 'count']
                    # pie chart
                    fig = px.pie(client_gender, values='count', names='Gender', hover_name='Gender', color_discrete_sequence =['steelblue','skyblue'])
                    fig.update_layout(showlegend=False,
                                      width=500,
                                      height=500,
                                      margin=dict(l=1, r=1, b=1, t=1),
                                      font=dict(color='#383635', size=15))

                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    st.write(fig)

                elif feature == 'Family status':
                    st.write('<u>Family status</u>', unsafe_allow_html=True)
                    # preparing data to display on pie chart
                    client_status = data_exploration['NAME_FAMILY_STATUS'].value_counts().reset_index()
                    client_status.columns = ['Family status', 'count']
                    # pie chart
                    fig = px.pie(client_status, values='count', names='Family status', hover_name='Family status', color_discrete_sequence =['steelblue','skyblue'])

                    fig.update_layout(showlegend=False,
                                      width=500,
                                      height=500,
                                      margin=dict(l=1, r=1, b=1, t=1),
                                      font=dict(color='#383635', size=15))

                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    st.write(fig)

                elif feature == 'Education type':
                    st.write('<u>Clients\' level of education</u>', unsafe_allow_html=True)
                    sns.histplot(data_exploration['NAME_EDUCATION_TYPE'], edgecolor='k', color='steelblue')
                    plt.title('Clients\' level of education ', size=9)
                    plt.xlabel('Level of education', size=9)
                    plt.ylabel('Frequency', size=9)
                    plt.xticks(rotation=90, size=9)
                    plt.yticks(size=9)
                    st.pyplot(fig)

                elif feature == 'Income type':
                    st.write('<u>Type of client income</u>', unsafe_allow_html=True)
                    sns.histplot(data_exploration['NAME_INCOME_TYPE'], edgecolor='k', color='steelblue')
                    plt.title('Type of client income ', size=9)
                    plt.xlabel('Income type', size=9)
                    plt.ylabel('frequency', size=9)
                    plt.xticks(rotation=90)
                    plt.xticks(size=9)
                    plt.yticks(size=9)
                    st.pyplot(fig)

                elif feature == 'Total income amount':
                    st.write('<u>Total income amount</u>', unsafe_allow_html=True)
                    sns.histplot(data_exploration['AMT_INCOME_TOTAL'], bins=70, color='steelblue')
                    plt.title('Distribution of clients by total income amount', size=9)
                    plt.xlabel('income (euros)', size=9)
                    plt.ylabel('Frequency', size=9)
                    plt.xticks(size=9)
                    plt.yticks(size=9)
                    plt.xlim([0, 6e5])
                    st.pyplot(fig)

                elif feature == 'Credit amount':
                    st.write('<u>Client\'s credit amount</u>', unsafe_allow_html=True)
                    sns.histplot(data_exploration['AMT_CREDIT'], bins=50, color='steelblue')
                    plt.title('Distribution of clients by credit amount', size=9)
                    plt.xlabel('Credit amount (euros)', size=9)
                    plt.ylabel('Frequency', size=9)
                    plt.xticks(size=9)
                    plt.yticks(size=9)
                    plt.xlim([1e4, 2e6])
                    st.pyplot(fig)

                elif feature == 'Contract type':
                    st.write('<u>Contract type</u>', unsafe_allow_html=True)
                    # preparing data to display on pie chart
                    contract_type = data_exploration['NAME_CONTRACT_TYPE'].value_counts().reset_index()
                    contract_type.columns = ['Contract type', 'count']
                    # pie chart
                    fig = px.pie(contract_type, values='count', names='Contract type', hover_name='Contract type', color_discrete_sequence =['steelblue','skyblue'])

                    fig.update_layout(showlegend=False,
                                      width=500,
                                      height=500,
                                      margin=dict(l=1, r=1, b=1, t=1),
                                      font=dict(color='#383635', size=15))

                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    st.write(fig)

                elif feature == "Annuity amount":
                    st.write('<u>Loan annuity</u>', unsafe_allow_html=True)
                    sns.histplot(data_exploration['AMT_ANNUITY'], bins=50, color='steelblue')
                    plt.title('Distribution of clients by loan annuity', size=9)
                    plt.xlabel('Annuity amount (euros)', size=9)
                    plt.ylabel('Frequency', size=9)
                    plt.xticks(size=9)
                    plt.yticks(size=9)
                    plt.xlim([1e3, 8e4])
                    st.pyplot(fig)

        elif visualization == "Bivariate analysis":
            # exploration bivariée des variables
            score = []
            df = pd.DataFrame()
            id_client_train = data_train_app["SK_ID_CURR"].values
            id_client_train = pd.DataFrame(id_client_train)
            for id in id_client_train.loc[:, 0]:
                idx = data_train_app.loc[data_train_app["SK_ID_CURR"] == float(id)].index
                data_train_client = x_train.iloc[idx]
                prediction = model_LR.predict_proba(data_train_client)
                prediction = prediction[0].tolist()
                score.append(prediction[1])
                data_train_client['score'] = prediction[1]
                df = df.append(data_train_client, ignore_index=True)
            df["SK_ID_CURR"] = id_client_train.loc[:, 0]

            st.markdown('#### Bivariate analysis of features :')
            if st.checkbox("Bivariate analysis (customers\' income - credit amount)"):
                st.write('<u> Customers\' income - credit amount </u>', unsafe_allow_html=True)
                fig = plt.figure(figsize=(8, 6))
                ax = plt.scatter(x=data_exploration['AMT_INCOME_TOTAL'], y=data_exploration['AMT_CREDIT'],
                                 c=df['score'] * 100, cmap='Oranges')
                norm = plt.Normalize(df['score'].min() * 100, df['score'].max() * 100)
                sm = plt.cm.ScalarMappable(cmap='Oranges', norm=norm)
                sm.set_array([])
                ax.figure.colorbar(sm)
                #plt.title('Credit amount according to clients\' income', size=9)
                plt.xlabel('Income (euros)', size=9)
                plt.ylabel('Credit amount (euros)', size=9)
                plt.xticks(size=9)
                plt.yticks(size=9)
                plt.ylim([1e4, 2e6])
                plt.xlim([3e4, 6e5])
                st.pyplot(fig)

            if st.checkbox("Bivariate analysis of clients' income and age"):
                st.write('<u> Clients\' income and age </u>', unsafe_allow_html=True)
                fig = plt.figure(figsize=(8, 6))
                ax = plt.scatter(x=data_exploration['DAYS_BIRTH'] / -365, y=data_exploration['AMT_INCOME_TOTAL'],
                                 c=df['score'] * 100, cmap='Oranges')
                norm = plt.Normalize(df['score'].min() * 100, df['score'].max() * 100)
                sm = plt.cm.ScalarMappable(cmap='Oranges', norm=norm)
                sm.set_array([])
                ax.figure.colorbar(sm)
                #plt.title(' Clients\' income and age', size=5)
                plt.xlabel('Age (years)', size=9)
                plt.ylabel('Income (euros)', size=9)
                plt.xticks(size=9)
                plt.yticks(size=9)
                plt.ylim([1e5, 8e5])
                st.pyplot(fig)

            if st.checkbox("Bivariate analysis (Years of employment - age of clients)"):
                st.write('<u>Age and working years of clients </u>', unsafe_allow_html=True)
                fig = plt.figure(figsize=(8, 6))
                ax = plt.scatter(x=data_exploration['DAYS_BIRTH'] / -365, y=data_exploration['DAYS_EMPLOYED'] / -365,
                                 c=df['score'] * 100, cmap='Oranges')
                norm = plt.Normalize(df['score'].min() * 100, df['score'].max() * 100)
                sm = plt.cm.ScalarMappable(cmap='Oranges', norm=norm)
                sm.set_array([])
                ax.figure.colorbar(sm)
                #plt.title('Age and years of employment of clients', size=5)
                plt.xlabel('Age (years)', size=9)
                plt.ylabel('Working years (years)', size=9)
                plt.xticks(size=9)
                plt.yticks(size=9)
                plt.ylim([0, 30])
                st.pyplot(fig)

# Affichage d'informations du client dans la sidebar
st.sidebar.header(":memo: Main customer information")

id_client = st.sidebar.selectbox("Select ID client", id_clients)
data_client = data_test_app[data_test_app["SK_ID_CURR"] == id_client]
idx = data_test_app.loc[data_test_app["SK_ID_CURR"] == id_client].index
data_selected_client = x_test.iloc[idx]

st.sidebar.write("**The ID client is :**", int(id_client))
infos_client = data_client_infos[data_client_infos['SK_ID_CURR'] == id_client].reset_index()
st.sidebar.write("**Age of client :**", int(infos_client['DAYS_BIRTH'].values /-365), "years")
if infos_client['CODE_GENDER'][0]=='M':
    st.sidebar.write("**Gender of client :**",  "Male ")
else:
    st.sidebar.write("**Gender of client :**", "Female")
st.sidebar.write("**Family status :**", infos_client["NAME_FAMILY_STATUS"][0])
st.sidebar.write("**Education level:**", infos_client["NAME_EDUCATION_TYPE"][0])
st.sidebar.write("**Income type :**", infos_client["NAME_INCOME_TYPE"][0])
st.sidebar.write("**Income amount:**", int(infos_client["AMT_INCOME_TOTAL"].values ), "euros")
if int(infos_client["DAYS_EMPLOYED"].values /-365)<=0 or int(infos_client["DAYS_EMPLOYED"].values /-365)>100:
    st.sidebar.write("**Year of work :**",  "unknown information ")
else:
    st.sidebar.write("**Year of work:**", int(infos_client["DAYS_EMPLOYED"].values /-365), "years")
st.sidebar.write("**Contract type :**", infos_client["NAME_CONTRACT_TYPE"][0])
st.sidebar.write("**Credit amount :**", int(infos_client["AMT_CREDIT"].values ), "euros")
st.sidebar.write("**Annuity amount:**", int(infos_client["AMT_ANNUITY"].values ), "euros")



with model_trainig:
    st.header('Analysis of the selected client\'s file')
    st.write("You have selected the customer :", int(id_client))
    st.markdown("<u>Selected client\'s data : </u>", unsafe_allow_html=True)
    st.write(data_client)
    # Affichage la solvabilité du client
    st.markdown("<u>Probability of customer bankruptcy risk :</u>", unsafe_allow_html=True)
    #prediction = load_prediction()
    prediction = model_LR.predict_proba(data_selected_client)
    target_client = model_LR['regressor'].predict(data_selected_client)
    score = prediction[0].tolist()
    risque = round(score[1] * 100, 2)
    st.write(risque, "%")
    threshold = st.slider("Select the threshold", 0, 100, 40)
    fig = plt.figure(figsize=(6, 6))
    fig = go.Figure(go.Indicator(domain={'x': [0, 1], 'y': [0, 1]},
                                 value=risque,
                                 mode="gauge+number",
                                 title={'text': "Probability of bankruptcy"},
                                 gauge={'axis': {'range': [None, 100]},
                                        'bar': {'color': "Saddlebrown", "thickness": 0.05},
                                        'steps': [
                                            {'range': [0, 20], 'color': "lightgoldenrodyellow"},
                                            {'range': [20, 40], 'color': "moccasin"},
                                            {'range': [40, 60], 'color': "lightcoral"},
                                            {'range': [60, 80], 'color': "indianred"},
                                            {'range': [80, 100], 'color': "darkred"}],

                                        'threshold': {'line': {'color': "red", 'width': 8}, 'thickness': 0.85,
                                                      'value': threshold}}))

    st.plotly_chart(fig)
    if risque <= 20:
        st.success('Very low bankruptcy risk: credit accepted')
    elif risque <= 40:
        st.success('Low bankruptcy risk: credit accepted')
    elif risque <= 60:
        st.warning('Medium bankruptcy risk: credit refused')
    elif risque <= 80:
        st.warning('Hight bankruptcy risk: credit refused')
    else:
        st.warning('Very hight bankruptcy risk: credit refused')


with model_explainer:
    st.header('Model explanation ')
    st.markdown("<u>Interpretation of the model - Local and global importance of features :</u>", unsafe_allow_html=True)
    #number = st.slider("Select the number of features...", 5, 100, 10)
    x_test_std = model_LR['scaler'].transform(x_test)
    # load the pickle shap explainer
    # shap_explainer = pickle.load(open("explainer//LR_shap_values.pkl", "rb"))
    #LR_explainer = pickle.load(open("models//LR_explainer.pkl", "rb"))
    LR_explainer = shap.LinearExplainer(model_LR['regressor'], x_test_std)
    LR_shap_values = LR_explainer(x_test_std)

    if st.checkbox("Local interpretation of the model"):
        number_local = st.slider("Select the number of features...", 5, 100, 10, key=1)
        data_selected_client_std = model_LR['scaler'].transform(data_selected_client)
        x_local = pd.DataFrame(data_selected_client_std, columns=x_test.columns)
        client_shap_values = LR_explainer(x_local)
        fig = plt.figure(figsize=(12, 12))
        shap.plots.waterfall(client_shap_values[0], max_display=number_local)
        st.pyplot(fig)

    if st.checkbox("Global interpretation of the model"):
        number_global = st.slider("Select the number of features...", 5, 100, 10, key=2)
        x_global = pd.DataFrame(x_test_std, columns=x_test.columns)
        fig = plt.figure(figsize=(12, 12))
        shap.summary_plot(LR_shap_values, x_global, plot_type="bar", max_display=number_global)
        st.pyplot(fig)

with similar_clients:
    st.header('Similar clients ')
    # Affichage des dossiers similaires
    similar_clients_10 = st.checkbox("Show clients with similar cases")

    if similar_clients_10:
        st.markdown("<u>List of the 10 files closest to this client :</u>", unsafe_allow_html=True)
        distances, indices = model_knn['regressor'].kneighbors(data_selected_client)
        similar_id = data_exploration.iloc[indices[0], :]
        st.write(similar_id)
        st.markdown("<i>**<u>Target = 1 :</u>** Credit refused, customer with bankruptcy risk </i>", unsafe_allow_html=True)


