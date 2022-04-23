
import streamlit as st

import pandas as pd
import numpy as np
from PIL import Image
import datetime

df = pd.read_csv('W_Aus_Na_mean.csv', index_col = 0)

selection = st.sidebar.radio(
    "Choix :",
    ("Description", "Réaliser une prédiction")
)

if selection == 'Description':
    st.markdown("<h1 style='text-align: center; color: black;'>Ok Python : do I need to take my umbrella tomorrow ?</h1>", unsafe_allow_html=True)
    st.write("-------------------------------------------------")

    st.markdown("<p style='text-align: center; color: black;'> Bienvenue sur cette apllication.</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: black;'> Son but est de prédire la probabilité de pluie du lendemain pour \
                un habitant australien.</p>", unsafe_allow_html=True)

    image = Image.open('drapeau.png')

    # Pour centrer l'image
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(' ')
    with col2:
        st.image(image, caption='Australie')
    with col3:
        st.write(' ')

    st.write("-------------------------------------------------")

    st.markdown("<h1 style='text-align: center; color: black;'>Comment fait on cela ?</h1>", unsafe_allow_html=True)
    st.markdown('''On dispose pour cette tâche d'une base de données labelisée : 
* On entraine un modèle de machine learning de classification
* On réalise la prédiction.
        ''')

    st.write("-------------------------------------------------")

    st.markdown("<h1 style='text-align: center; color: black;'>De quelles données dispose-t-on ?</h1>", unsafe_allow_html=True)
    
elif selection == 'Réaliser une prédiction':
    st.markdown("<h1 style='text-align: center; color: black;'>Ok Python : do I need to take my umbrella tomorrow ?</h1>", unsafe_allow_html=True)
    st.write("-------------------------------------------------")
    
    st.markdown("<p style='text-align: center; color: black;'> Nous allons vous aider à réaliser une prédiction.</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: black;'> Pour cela, voudriez vous bien nous fournir quelques informations \
     dans la section suivante ? </p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: black;'> Merci :) </p>", unsafe_allow_html=True)
    
    st.write("-------------------------------------------------")
    
    st.markdown("<h1 style='text-align: center; color: black;'>Données météorologiques du jour</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: left; color: black;'> Pour nous permettre de réaliser la prédiction, \
    merci de renseigner les données météorologiques suivantes pour la journée en cours :</p>", unsafe_allow_html=True)
    
    Date = st.date_input('Date', datetime.date(2022, 4, 27))
    Location = st.selectbox('Localisation',  tuple(df["Location"].sort_values().unique())   )
    WindDir9am = st.selectbox('Direction moyenne du vent entre 8h50 et 9h', tuple(df["WindDir9am"].sort_values().unique())    )
    WindDir3pm = st.selectbox('Direction moyenne du vent entre 14h50 et 15h', tuple(df["WindDir3pm"].sort_values().unique())   )
    WindGustDir = st.selectbox('Direction de la plus forte rafale de vent', tuple(df["WindGustDir"].sort_values().unique()) )
    WindGustSpeed = st.slider("Vitesse de la plus forte rafale de vent (en km/h)", 
                              int(df["WindGustSpeed"].min()), int(df["WindGustSpeed"].max()), int(df["WindGustSpeed"].mean())) 
    MinTemp = st.slider('Température minimale', -20, 100, 10) 
    MaxTemp = st.slider('Température maximale', 0, 100, 40) 
    Rainfall = st.slider('Précipitations (en mm)', 0, 20, 0) 
    Evaporation = st.slider('Evaporation (en mm)', 0, 20, 0)  
    Sunshine = st.slider("Nombre d'heures d'ensoleillement", 0, 24, 5)  
    WindSpeed9am = st.slider("Vitesse moyenne du vent entre 8h50 et 9h (en km/h)", 0, 100, 5)  
    WindSpeed3pm = st.slider("Vitesse moyenne du vent entre 14h50 et 15h (en km/h)", 0, 100, 5)  
    Humidity9am = st.slider("Taux d'humidité à 9h (en %)", 0, 100, 5)
    Humidity3pm = st.slider("Taux d'humidité à 15h (en %)", 0, 100, 5)
    Pressure9am = st.slider("Pression atmosphérique à 9h(en hectopascals)", 500, 2000, 1000) 
    Pressure3pm = st.slider("Pression atmosphérique à 15h(en hectopascals)", 500, 2000, 1000) 
    Cloud9am = st.select_slider('Fraction du ciel obscurcie par les nuages à 9h',
     options=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])  
    Cloud3pm = st.select_slider('Fraction du ciel obscurcie par les nuages à 15h',
     options=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])  
    Temp9am = st.slider("Température à 9h", -20, 80, 20) 
    Temp3pm = st.slider("Température à 15h", -20, 80, 20) 
    RainToday = st.radio("Est ce qu'il a plu aujourd'hui :",
    ("Yes", "No")
)

    st.write("-------------------------------------------------")
    
    st.markdown("<h1 style='text-align: center; color: black;'>Prévision de pluie pour le lendemain</h1>", unsafe_allow_html=True)
    modele = st.radio("Choisir votre modèle de prédiction :",
    ("Regression logistique", "KNN", "SVM", "RandomForest", "XGboost", "Réseau de neurones dense"))
