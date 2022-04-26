
import weakref
import streamlit as st
import io
import pandas as pd
import numpy as np
from PIL import Image
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix
from imblearn.metrics import classification_report_imbalanced
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from bokeh.io import output_file, show,output_notebook
from bokeh.plotting import figure, output_notebook, show
from bokeh.models import ColumnDataSource, GMapOptions, HoverTool
from bokeh.plotting import gmap
import xgboost as xgb

df_mean = pd.read_csv('W_Aus_Na_mean.csv', index_col = 0)
df = pd.read_csv('weatherAUS.csv')
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

    #image = Image.open('drapeau.png')

    # Pour centrer l'image
    #col1, col2, col3 = st.columns(3)
    #with col1:
        #st.write(' ')
    #with col2:
        #st.image(image, caption='Australie')
    #with col3:
        #st.write(' ')

    st.write("-------------------------------------------------")
    
    st.markdown("<h1 style='text-align: center; color: black;'>Comment fait on cela ?</h1>", unsafe_allow_html=True)
    st.markdown('''On dispose pour cette tâche d'une base de données labelisée : 
* On entraine un modèle de machine learning de classification
* On réalise la prédiction.
        ''')

    st.write("-------------------------------------------------")

    
    st.markdown("<h1 style='text-align: center; color: black;'>De quelles données dispose-t-on ?</h1>", unsafe_allow_html=True)
    st.markdown("<h0 style='text-align: center; color: black;'>Le dataset weatherAUS.csv contient des données météorologiques issues des observations de nombreuses stations météo réparties sur l’ensemble du territoire Australien. Le dataset téléchargeable sur le site internet Kaggle, compile environ 10 ans d’observations quotidiennes fournies par le bureau of meteorology du gouvernement australien (http://www.bom.gov.au/climate/data/), qui met à disposition de nombreuses données en libre accès...</h0>", unsafe_allow_html=True)
    st.dataframe(df)

    st.markdown("""<h0 style='text-align: center; color: black;'>Il est composé de 23 variables, pour lesquelles ont peut constater à première vue
    qu’il y a de nombreuses valeurs manquantes, et que plusieurs sont de type catégorielles.</h0>""", unsafe_allow_html=True)
    
    # création de 2 colonnes pour afficher des informations concernant le dataset
    col1,col2 = st.columns(2)
    with col1:
        #affichage du % de valeurs manquantes
        st.table(pd.DataFrame(round(df.isnull().sum()/(df.shape[0])*100,2),
        columns=['% valeurs manquantes']).sort_values('% valeurs manquantes',ascending=False))
    with col2:
        #affichage de df.info() pour cela il faut changer le format de sortie 
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)
    st.markdown("""<h0 style='text-align: center; color: black;'>Les variables “Cloud3pm”, “Sunshine” et “Evaporation” ont plus de 40% de valeurs manquantes.
    Quant à la variable “Clound9am” elle dénombre 38% de valeurs manquantes. Nous décidons de supprimer les variables dont plus de 40% des valeurs sont manquantes. 
    En effet, nous avons constaté aux cours des différentes itérations que nos modèles prédictifs étaient plus robustes, sans ces variables. 
    L’imputation d’un grand nombre de valeurs en remplacement des NaN a une incidence sur la qualité de l’apprentissage de nos modèles.</h0>""", unsafe_allow_html=True)

    #création d'une carte avec la position de chaque point de la variable "Location" du dataset

    #importation du fichier CSV contenant les données géographiques des villes du Dataset "weatherAUS".
    #les variables latitude et longitude ont été récupérées en connectant à l'API de googlemap
    df_ville = pd.read_csv("geocoordonnees.csv",index_col=0)
    #définition des variables correspondant à la latitude et la longitude de l'Australie
    lat = -27.157944693150345
    lng = 133.55059052037544
    map_options = GMapOptions(lat=lat, lng=lng, map_type="terrain", zoom=4)
    p =gmap("AIzaSyC989eZT8qV1z5p3LqYpGa1KkwuqCLucJM", map_options, title="Localisation des différentes villes du dataset")
    source = ColumnDataSource(data=df_ville)
    c= p.circle(x='Longitude', y='Latitude', size=8, fill_color="red", fill_alpha=0.8,source=source)
    #faire apparaitre le nom de la ville lorsque que le curseur passe au dessus d'un point
    tooltips = "@Location"
    hover = HoverTool(tooltips = tooltips, renderers = [c])

    p.add_tools(hover)
    #affichage de la carte dans streamlit
    with st.expander("Cartographie des villes du Dataset"):
        st.markdown("""<h0 style='text-align: center; color: black;'> En s'appuyant sur l'API de googlemap, nous obtenons les coordonnées géographiques de chaque ville du dataset afin de pouvoir les cartographier.</h0>""", unsafe_allow_html=True)
        st.bokeh_chart(p,use_container_width=False)
    
    #affichage de la carte météo de l'Australie
    with st.expander("Climats australiens selon la classification de Köppen"):
        st.markdown("""<h0 style='text-align: center; color: black;'>Bien que majoritairement aride, l'Australie est un pays immense dont le climat varie d'une région à une autre.</h0>""", unsafe_allow_html=True)
        image_2 = Image.open('australia_meteo_map.png')
        st.image(image_2,caption="Climats d'Australie")


    #remplacement des modalités des variables 'RainTomorrow' et 'RainToday' par 1 ou 0
    df2=pd.read_csv("W_Aus_Na_mean.csv",index_col=0)
    df2['RainTomorrow']=df2['RainTomorrow'].replace({'No':0,'Yes':1})
    df2['RainToday']=df2['RainToday'].replace({'No':0,'Yes':1})
    # création d'une liste des variable catégorielle
    l = []
    for i in df2.columns:
        if df2.dtypes[i]=='O':
            l.append(i)
    # encoder les variables catégorielle avec la classe LebelEncoder
    la = LabelEncoder()
    for i in l:
        df2[i] = la.fit_transform(df2[i])
    # Correlations entre variables numériques
    data = round(df2.iloc[:,1:23].corr().abs(),2) #on ne représente pas les variables que nous avons créées (State,day,year...ni date)
    #réalisation d'une heatmap de la matrice de corrélation
    fig, ax = plt.subplots(figsize=(17,13))
    sns.heatmap(data,ax=ax,annot = True, cmap = "Spectral" )
    #affichage de la heatmap
    with st.expander("Analyse de la corrélation linéaire des variables - Coef de Pearson"):
        col1,col2 = st.columns(2)
        with col1:
            st.pyplot(fig)
        # Tri par ordre décroissant des valeurs aboslues du coefficient de pearson
        with col2:
            related = data['RainTomorrow'].sort_values(ascending = False)
            related
        st.markdown("""<h0 style='text-align: center; color: black;'>Plusieurs variables présentent une corrélation linéaire significative avec la variable cible “RainTomorrow”,
         telles que “Humidity3pm”,”Sunshine” ou encore les variables “Cloud3pm” et “Cloud9am”.</h0>""", unsafe_allow_html=True)


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
