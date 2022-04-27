
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
from joblib import dump, load
from keras.models import load_model 
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
    st.markdown("<h1 style='text-align: center; color: black;'>Que nous racontent ces données ?</h1>", unsafe_allow_html=True)
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

    with st.expander("Distribution de la variable cible"):    
        col1,col2 = st.columns(2)
        with col1:
            # distribution de la variable cible
            fig = plt.figure()
            ax = fig.add_subplot(111)
            sns.countplot(ax = ax, x = df["RainTomorrow"])
            plt.title('Distribution de la variable RainTomorrow')
            st.pyplot(fig)
        with col2:
            # Aperçu de l'équilibre des classes
            st.table(pd.DataFrame((df["RainTomorrow"].value_counts(normalize = True)*100).reset_index().rename(columns={'index':'RainTomorrow','RainTomorrow':'Fréquence en %'})))
        st.markdown("""<h0 style='text-align: center; color: black;'>Il y a un déséquilibre de classe important de la variable cible RainTomorrow, qui dénombre 78% de jours   sans pluie contre 22% 
        de jours de pluie soit quasiment un rapport de 1 pour 4. Pour éviter le surapprentissage avec notre
        modèle de prédiction, nous allons rééquilibrer les classes par Oversampling</h0>""", unsafe_allow_html=True)
        fig = plt.figure()
        # Histo - distribution de la variable cible en fonction de RainToday
        ax = fig.add_subplot(111)
        sns.countplot(df.RainTomorrow,ax=ax,hue = df.RainToday)
        plt.title('Distribution de la variable RainTomorrow en fonction de RainToday')
        st.pyplot(fig)
        st.markdown("""<h0 style='text-align: center; color: black;'>Lorsqu'il pleut le jour J (RainToday),
         la probabilité qu'il pleuve le lendemain (RainTomorrow) est beaucoup plus importante.</h0>""", unsafe_allow_html=True)
        

        st.markdown("""<h5 style='text-align: center; color: black;'>Distribution de RainTomorrow par Etat</h5>""", unsafe_allow_html=True)
        # Histo - distribution de la varaible cible par Etat
        # calcul de la fréquence de la variable RainTomorrow par état
        df_frequency = pd.DataFrame(df_mean.groupby(['State','RainTomorrow']).size())
        df_frequency.reset_index(inplace=True)
        df_frequency['freq']=1
        for s in (df_frequency['State']):
            for r,i in zip (df_frequency[df_frequency['State']==s]['RainTomorrow'],list(df_frequency[df_frequency['State']==s].index)):
                f = df_frequency[df_frequency['State']==s][0].sum()
                df_frequency.iloc[i,3]= df_frequency[(df_frequency['State']==s)&(df_frequency['RainTomorrow']==r)][0]/f
        
        f = plt.figure(figsize=(16, 9.5))
        gs = f.add_gridspec(2, 4)
        ax = f.add_subplot(gs[0, 0])
        sns.barplot(x='RainTomorrow',y='freq',data=df_frequency[df_frequency['State']=="South Australia"])
        plt.title('South Australia')
        plt.ylim(0,0.9)
        ax = f.add_subplot(gs[0, 1])
        sns.barplot(x='RainTomorrow',y='freq',data=df_frequency[df_frequency['State']=="New South Wales"])
        plt.title('New South Wales')
        plt.ylim(0,0.9)
        ax = f.add_subplot(gs[0, 2])
        sns.barplot(x='RainTomorrow',y='freq',data=df_frequency[df_frequency['State']=="NorfolkIsland"])
        plt.title('NorfolkIsland')
        plt.ylim(0,0.9)
        ax = f.add_subplot(gs[0, 3])
        sns.barplot(x='RainTomorrow',y='freq',data=df_frequency[df_frequency['State']=="Northern Territory"])
        plt.title('Northern Territory')
        plt.ylim(0,0.9)
        ax = f.add_subplot(gs[1, 0])
        sns.barplot(x='RainTomorrow',y='freq',data=df_frequency[df_frequency['State']=="Queensland"])
        plt.title('Queensland')
        plt.ylim(0,0.9)
        ax = f.add_subplot(gs[1, 1])
        sns.barplot(x='RainTomorrow',y='freq',data=df_frequency[df_frequency['State']=="Tasmanie"])
        plt.title('Tasmanie')
        plt.ylim(0,0.9)
        ax = f.add_subplot(gs[1, 2])
        sns.barplot(x='RainTomorrow',y='freq',data=df_frequency[df_frequency['State']=="Victoria"])
        plt.title('Victoria')
        plt.ylim(0,0.9)
        ax = f.add_subplot(gs[1, 3])
        sns.barplot(x='RainTomorrow',y='freq',data=df_frequency[df_frequency['State']=="Western Australia"])
        plt.title('Western Australia')
        plt.ylim(0,0.9)
        st.pyplot(f)
        st.markdown("""<h0 style='text-align: center; color: black;'>La distribution de la variable cible entre les différents États d’Australie est relativement homogène.</h0>""", unsafe_allow_html=True)

        st.markdown("""<h5 style='text-align: center; color: black;'>Distribution de RainTomorrow en fonction de la direction du vent et de la couverture nuageuse</h5>""", unsafe_allow_html=True)
        #histo - distribution de la variable cible en fonction de WindGustDir,WinDir et Cloud
        f = plt.figure(figsize=(10, 10))
        gs = f.add_gridspec(3, 2)
        ax = f.add_subplot(gs[0, 0])
        sns.countplot(df.WindGustDir,ax=ax, palette = 'colorblind', hue = df.RainTomorrow)
        plt.xticks(size = 6)
        ax = f.add_subplot(gs[0, 1])
        sns.countplot(df.WindDir9am,ax=ax,palette = 'colorblind', hue = df.RainTomorrow)
        plt.xticks(size = 6)
        ax = f.add_subplot(gs[1, 0])
        sns.countplot(df.WindDir3pm,ax=ax,palette = 'colorblind', hue = df.RainTomorrow)
        plt.xticks(size = 6)
        ax = f.add_subplot(gs[1, 1])
        sns.countplot(df.Cloud9am,ax=ax, palette = 'colorblind', hue = df.RainTomorrow)
        plt.xticks(size = 6)
        ax = f.add_subplot(gs[2, 0])
        sns.countplot(df.Cloud3pm,ax=ax, palette = 'colorblind', hue = df.RainTomorrow)
        plt.xticks(size = 6)
        st.pyplot(f)
        st.markdown("""<h0 style='text-align: center; color: black;'>La distribution des variables liées à la direction du vent sont relativement homogène à l’exception du vent du Nord de la variable “WindDir9am” qui semble plus fortement corrélé
         au fait qu’il pleuve le lendemain (“RainTomorrow” = yes). En revanche les variables “Cloud3pm” et “Cloud9am” sont étroitement liées au temps qu’il fera le lendemain.</h0>""", unsafe_allow_html=True)

    with st.expander("Analyse de la pluviométrie"): 
        #Moyenne des chutes de pluie par an et par Etat
        f = sns.relplot(df_mean.year,df_mean.Rainfall,hue = df_mean.State, ci = None,kind = 'line',height=6,aspect=2)
        plt.ylabel('rainfall en mm')
        plt.title('Chutes de pluie par Etat')
        st.pyplot(f)
        

        st.markdown("""<h5 style='text-align: center; color: black;'>Saisonnalité - Moyenne mensuelle du volume de pluie</h5>""", unsafe_allow_html=True)
        #pluie en fonction des mois puis de l'année
        g=sns.relplot(df_mean.year_month,df_mean.Rainfall,col = df_mean.State,col_wrap=2, ci = None,kind = 'line',height=2,aspect=4)
        g.set_xlabels('de 2007 à 2017')
        plt.ylabel('rainfall en mm')
        plt.xticks("")
        st.pyplot(g)
        st.markdown("""<h0 style='text-align: center; color: black;'>L’évolution de la pluviométrie semble globalement constante avec quelques variations ponctuelles, 
        dont l’explication peut se trouver dans les phénomènes météorologiques auxquels est confrontée l’Australie tels que La Niña et El Niño.  
        Il y a notamment eu un épisode  La Niña particulièrement virulent au dernier trimestre 2010, qui s’est caractérisé par d’importantes chutes de pluie dans l’Est de l’Australie
         et des sécheresses dans l’Ouest. Ce phénomène est parfaitement représenté ici avec une moyenne de pluie particulièrement élevée pour le Queensland située à l’Est 
         de l’Australie et inversement une moyenne particulièrement faible pour le Western Australia.</h0>""", unsafe_allow_html=True)

    with st.expander("Identification des valeurs extrêmes"):
        st.markdown("""<h5 style='text-align: center; color: black;'>Saisonnalité - Moyenne mensuelle du volume de pluie</h5>""", unsafe_allow_html=True)
        #Création de nouvelles colonnes contenant les mois, jours, années et année + mois car ça nous sera util pour l'exploration
        #Définition des fonctions à appliquer à la colonne 'Date'
        def get_day(date):
            splits = date.split('-')    
            day = splits[2]
            return day

        def get_month(date):
            return date.split('-')[1]

        def get_year(date):
            return date.split('-')[0]   
        # Application des fonctions
        days = df['Date'].apply(get_day)
        months = df['Date'].apply(get_month)
        years = df['Date'].apply(get_year)
        # Création des nouvelles colonnes
        df['day'] = days
        df['month'] = months
        df['year'] = years
        df['year_month']= years+"-"+months
        # changement de type de donnée des colonnes month, day, et year
        df = df.astype({'year':'int64','month':'int64','day':'int64'})
        #importation du fichier CSV contenant les variables qui nous intéressent
        states = pd.read_csv('australian_states.csv',';',index_col=0)
        #fusion des deux DataFrames
        df = pd.merge(df,states)
        #on fait pareil avec le fichier csv que nous avons crée contenant les coordonées géo des villes
        lat_long = pd.read_csv("geocoordonnees.csv",index_col=0)
        df = pd.merge(df,lat_long)

        fig = plt.figure(figsize=(20, 17))
        gs = fig.add_gridspec(2, 2)

        ax = fig.add_subplot(gs[0, 0])
        sns.boxplot(y='MaxTemp',x='State',data=df,ax=ax)
        sns.despine(top = True, bottom = True, left = False, right = False)
        plt.xticks(rotation=20)
        ax = fig.add_subplot(gs[0, 1])
        sns.boxplot(y='MinTemp',x='State',data=df,ax=ax)
        sns.despine(top = True, bottom = True, left = False, right = False)
        plt.xticks(rotation=20)
        ax = fig.add_subplot(gs[1, 0])
        sns.boxplot(y='Temp9am',x='State',data=df,ax=ax)
        sns.despine(top = True, bottom = True, left = False, right = False)
        plt.xticks(rotation=20)
        ax = fig.add_subplot(gs[1, 1])
        sns.boxplot(y='Temp3pm',x='State',data=df,ax=ax)
        sns.despine(top = True, bottom = True, left = False, right = False)
        plt.xticks(rotation=20)
        st.pyplot(fig)

        fig = plt.figure(figsize=(20, 25))
        gs = fig.add_gridspec(4, 3)
        ax = fig.add_subplot(gs[0, 0])
        sns.boxplot(y='Humidity3pm',x='State',data=df,ax=ax)
        sns.despine(top = True, bottom = True, left = False, right = False)
        plt.xticks(rotation=20)
        ax = fig.add_subplot(gs[0, 1])
        sns.boxplot(y='Cloud9am',x='State',data=df,ax=ax)
        sns.despine(top = True, bottom = True, left = False, right = False)
        plt.xticks(rotation=20)
        ax = fig.add_subplot(gs[0, 2])
        sns.boxplot(y='Cloud3pm',x='State',data=df,ax=ax)
        sns.despine(top = True, bottom = True, left = False, right = False)
        plt.xticks(rotation=20)
        ax = fig.add_subplot(gs[1, 0])
        sns.boxplot(y='Rainfall',x='State',data=df,ax=ax)
        sns.despine(top = True, bottom = True, left = False, right = False)
        plt.xticks(rotation=20);
        ax = fig.add_subplot(gs[1, 1])
        sns.boxplot(y='Evaporation',x='State',data=df,ax=ax)
        sns.despine(top = True, bottom = True, left = False, right = False)
        plt.xticks(rotation=20)
        ax = fig.add_subplot(gs[1, 2])
        sns.boxplot(y='Humidity9am',x='State',data=df,ax=ax)
        sns.despine(top = True, bottom = True, left = False, right = False)
        plt.xticks(rotation=20)
        ax = fig.add_subplot(gs[2, 0])
        sns.boxplot(y='Pressure9am',x='State',data=df,ax=ax)
        sns.despine(top = True, bottom = True, left = False, right = False)
        plt.xticks(rotation=20)
        ax = fig.add_subplot(gs[2, 1])
        sns.boxplot(y='Pressure3pm',x='State',data=df,ax=ax)
        sns.despine(top = True, bottom = True, left = False, right = False)
        plt.xticks(rotation=20)
        ax = fig.add_subplot(gs[3, 0])
        sns.boxplot(y='WindGustSpeed',x='State',data=df,ax=ax)
        sns.despine(top = True, bottom = True, left = False, right = False)
        plt.xticks(rotation=20)
        ax = fig.add_subplot(gs[3, 1])
        sns.boxplot(y='WindSpeed9am',x='State',data=df,ax=ax)
        sns.despine(top = True, bottom = True, left = False, right = False)
        plt.xticks(rotation=20)
        ax = fig.add_subplot(gs[3, 2])
        sns.boxplot(y='WindSpeed3pm',x='State',data=df,ax=ax)
        sns.despine(top = True, bottom = True, left = False, right = False)
        plt.xticks(rotation=20)
        st.pyplot(fig)
        st.markdown("""<h0 style='text-align: center; color: black;'>Il y a de nombreuses valeurs extrêmes pour plusieurs variables, mais il est difficile de distinguer celles qui sont issues d’un évènement climatique isolé, à celles qui sont liées à des évènements climatiques cycliques tels que El Niño et La Ninã.
        Bien que ces valeurs extrêmes soient plausibles, il faudra les supprimer pour éviter de biaiser l’apprentissage de nos modèles de prédictions. Elles auraient pu être conservées, si le but avait été de prédire un phénomène météorologique rare conduisant à ce type de valeur.
        </h0>""", unsafe_allow_html=True)
    
    st.markdown("<h1 style='text-align: center; color: black;'>Comment avons-nous préparer les données pour notre modèle de prédiction ?</h1>", unsafe_allow_html=True)

    st.markdown('''<h0 style='text-align: center; color: black;'>L’exploration des données a montré que 7 étapes de prétraitement sont à minima nécessaires afin de disposer d’une base de données
 « propre » permettant un bon apprentissage des modèles de machine learning :
* La suppression des labels sans étiquettes
* La suppression des variables dont le nombre de valeurs manquantes est supérieur à 40%
* L’identification et la suppression des outliers
* Le traitement des valeurs manquantes de l’ensemble des variables,
* La transformation des variables catégorielles en variables indicatrices,
* La normalisation des données,
* Le rééquilibrage des classes.</h0>''', unsafe_allow_html=True)

    with st.expander("Suppressions des labels sans étiquettes"):
            st.markdown("""<h0 style='text-align: center; color: black;'>
            Dans un premier temps nous  supprimons les lignes dont les labels ne sont pas étiquetés.  
            Nous supprimons également les variables dont plus de 40% des valeurs sont manquantes. Les variables suivantes sont concernées: 'Evaporation','Sunshine','Cloud3pm'.</h0>""", unsafe_allow_html=True)
            code = '''# On défini un nouveau Dataframe en ne gardant dans un premier temps que les données étiquetées
            df_mean = df[(df["RainTomorrow"] == "No") | (df["RainTomorrow"] == "Yes") ]
            #suppression des variables pour lesquelles un trop grand nombre valeurs est manquant
            df_mean.drop(['Evaporation','Sunshine','Cloud3pm'],axis=1,inplace=True)'''
            st.code(code,language='python')


    with st.expander("Identification et suppression des outliers"):
         st.markdown("""<h0 style='text-align: center; color: black;'>Préalablement au traitement des valeurs manquantes, il faut supprimer les outliers qui peuvent avoir un impact sur les valeurs qui seront imputées.  
        Pour l’identification des outliers nous avons utilisé la méthode Inter Quartile Range (IQR) qui se calcul de la manière suivante:  
        IQR = Q3 - Q1  
        où :  
        Q1 est défini comme le nombre intermédiaire entre le plus petit nombre et la médiane de l’ensemble de données.  
        Q3 est la valeur médiane entre la médiane et la valeur la plus élevée de l’ensemble de données.  
        Avec l’IQR nous pouvons fixer la limite supérieure et inférieure d’un ensemble de valeurs pour une variable, de la manière suivante:  
        supérieur = Q3 +IQR  
        inférieur = Q1 – IQR  
        L’identification des outliers se fait pour chaque variable continue, mais en fonction de la variable ‘State’,
         car la zone géographique joue un rôle important sur les valeurs des variables.</h0>""", unsafe_allow_html=True)
 

    with st.expander("Imputation des valeurs manquantes"):
         st.markdown("""<h0 style='text-align: center; color: black;'>Le traitement des valeurs manquantes a fait l’objet de 2 stratégies :  
* Imputation avec la méthode KNNImputer de Sklearn : la valeur manquante d’une variable pour une observation donnée est remplacée par la moyenne de ses k plus proches voisins trouvés par l’algorithme k-Nearest Neighbors, les plus proches voisins étant les observations qui sont proches vis-à-vis des valeurs des autres variables non manquantes. Le nombre de voisins choisis est k = 2.  
* Imputation par la moyenne pour les variables numériques et par la modalité la plus fréquente pour les variables catégorielles.  
Chacune de ces méthodes de traitement fait l’objet d’une sauvegarde sous forme de fichier « csv », l’idée étant de pouvoir comparer l’impact de ces 2 stratégies sur les performances des modèles de machine learning testés plus tard en chargeant l’un ou l’autre des fichiers.</h0>""", unsafe_allow_html=True)


    with st.expander("Numérisation des variables"):
         st.markdown("""<h0 style='text-align: center; color: black;'>Afin de permettre la prise en compte des variables catégorielles dans l'entraînement de certains modèles (les SVM en particulier, la régression logistique
        ou encore l’algorithme des k plus proches voisins), il est nécessaire de transformer les modalités de celles-ci en valeurs numériques.  
        Pour cela, plusieurs possibilités existent en préprocessing des données :   

        * La méthode « get_dummies », qui permet de créer autant de colonnes que de modalités pour une seule variable, en y associant la valeur de 1 lorsque
         la modalité est présente, et 0 si non. Cette méthode n’a pas été retenue : en effet, on constate une augmentation significative des variables (de 27 à environ 120)  
        * La méthode  « LabelEncoder », qui permet d’assigner à chaque modalité d’une variable un entier compris entre 0 et le nombre de modalités de la variable
        moins 1. C’est cette méthode qui est retenue ici : elle a le mérite de ne pas conduire à une augmentation du nombre de variables.  
        A noter que la variable cible « RainTomorrow » est également encodée : la modalité « No » est remplacée par la valeur 0, et la modalité « Yes » par la valeur 1.</h0>""", unsafe_allow_html=True)



    with st.expander("Normalisation des données"):
         st.markdown("""<h0 style='text-align: center; color: black;'>Les variables à disposition présentent des plages de variation qui peuvent s’avérer assez éloignées. Comme le montre le graphique en boîtes à moustaches suivant, les variables de pression « Pressure9am » et « Pressure3pm » ont des plages de variations autour de 1000, tandis que pour le reste des variables, on est plutôt en moyenne autour de la dizaine, ou la centaine.  
        Un extrait de la méthode « describe » de Pandas, appliquée à notre DataFrame, permet de visualiser ces plages de variations :  </h0>""", unsafe_allow_html=True)
         st.markdown("""<h0 style='text-align: center; color: black;'>Afin de ne pas avoir de variable qui écrase les autres et qu’elle prenne un poids trop important dans la décision d’un modèle du fait de sa valeur (notamment en régression), on procède à une normalisation des données. On utilise pour cela la méthode « StandardScaler », qui consiste à soustraire la moyenne et à diviser par l’écart type l’ensemble des valeurs de la variable.   
        À l'issue de ce traitement, l’ensemble des variables explicatives de notre dataset sont de moyenne nulle et de variance 1.</h0>""", unsafe_allow_html=True)




    with st.expander("Rééquilibrage des classes"):
         st.markdown("""<h0 style='text-align: center; color: black;'>Comme vu précédemment, le nombre de données correspondant à la classe « Yes » de la variable cible
          « RainTomorrow » est nettement inférieur à la classe « No ». Si ce déséquilibre semble naturel et peut s’expliquer d’un point de vue météorologique (globalement sur une année, le nombre de jours sans pluie est souvent supérieur au nombre de jours avec pluie), il est en revanche problématique dans le cadre de l'entraînement des modèles de machine learning : un modèle pourra avoir un score en apparence élevé rien qu’en réalisant les bonnes prédictions sur la classe majoritaire.  
        Le rééquilibrage des classes est réalisé de plusieurs façons :  
        * Utilisation du paramètre « class_weight » associé aux modèles. L’intérêt est que cela ne nécessite pas d’action supplémentaire. En revanche cette option n’est pas
        disponible sur tous les modèles de machine learning  
        * Utilisation de la méthode SMOTE, qui est une méthode « d’over sampling » consistant à augmenter artificiellement le nombre d’observations de la classe minoritaire.
        Concrètement, on détermine les k plus proches voisins d’une observation minoritaire, puis on sélectionne l’un de ses k voisins, pour enfin créer une nouvelle observation située entre notre observation et le voisin sélectionné.  
        Typiquement, les algorithmes de Régression logistique, les SVM et le RandomForest embarquent le paramètre « class_weight », mais ce n’est pas le cas de l’algorithme
        de classification par les k plus proches voisins.
        </h0>""", unsafe_allow_html=True)




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
    
    df=pd.read_csv("W_Aus_Na_mean.csv",index_col=0)
    
    Date = st.date_input('Date', datetime.date(2022, 4, 27))
    Location = st.selectbox('Localisation',  tuple(df["Location"].sort_values().unique())   )
    WindDir9am = st.selectbox('Direction moyenne du vent entre 8h50 et 9h', tuple(df["WindDir9am"].sort_values().unique())    )
    WindDir3pm = st.selectbox('Direction moyenne du vent entre 14h50 et 15h', tuple(df["WindDir3pm"].sort_values().unique())   )
    WindGustDir = st.selectbox('Direction de la plus forte rafale de vent', tuple(df["WindGustDir"].sort_values().unique()) )
    WindGustSpeed = st.slider("Vitesse de la plus forte rafale de vent (en km/h)", 
                              0, 250, int(df["WindGustSpeed"].mean())) 
    MinTemp = st.slider('Température minimale', -20, 100, int(df["MinTemp"].mean())) 
    MaxTemp = st.slider('Température maximale', 0, 100, int(df["MaxTemp"].mean())) 
    Rainfall = st.slider('Précipitations (en mm)', 0, 100, int(df["Rainfall"].mean())) 
    Evaporation = st.slider('Evaporation (en mm)', 0, 100, int(df["Evaporation"].mean()))  
    Sunshine = st.slider("Nombre d'heures d'ensoleillement", 0, 24, int(df["Sunshine"].mean()))  
    WindSpeed9am = st.slider("Vitesse moyenne du vent entre 8h50 et 9h (en km/h)", 0, 200, int(df["WindSpeed9am"].mean()))  
    WindSpeed3pm = st.slider("Vitesse moyenne du vent entre 14h50 et 15h (en km/h)", 0, 200, int(df["WindSpeed3pm"].mean()))  
    Humidity9am = st.slider("Taux d'humidité à 9h (en %)", 0, 100, int(df["Humidity9am"].mean()))
    Humidity3pm = st.slider("Taux d'humidité à 15h (en %)", 0, 100, int(df["Humidity3pm"].mean()))
    Pressure9am = st.slider("Pression atmosphérique à 9h(en hectopascals)", 500, 2000, int(df["Pressure9am"].mean())) 
    Pressure3pm = st.slider("Pression atmosphérique à 15h(en hectopascals)", 500, 2000, int(df["Pressure3pm"].mean())) 
    Cloud9am = st.select_slider('Fraction du ciel obscurcie par les nuages à 9h',
     options=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])  
    Cloud3pm = st.select_slider('Fraction du ciel obscurcie par les nuages à 15h',
     options=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])  
    Temp9am = st.slider("Température à 9h", -20, 80, int(df["Temp9am"].mean())) 
    Temp3pm = st.slider("Température à 15h", -20, 80, int(df["Temp3pm"].mean())) 
    RainToday = st.radio("Est ce qu'il a plu aujourd'hui :", ("Yes", "No"))

    st.write("-------------------------------------------------")
    
    st.markdown("<h1 style='text-align: center; color: black;'>Prévision de pluie pour le lendemain</h1>", unsafe_allow_html=True)
    modele = st.radio("Choisir votre modèle de prédiction :",
    ("Regression logistique", "KNN", "SVM", "RandomForest", "XGboost", "Réseau de neurones dense"))
    # "GradientBoostingClassifier",
    
    clicked = st.button("Prédire")
    
    # Si il y a un clique
    if clicked:
        ##########################
        # Recharger les modèles 
        lr = load('classifieur_lr_mean.joblib')
        clf_knn = load('classifieur_knn_mean.joblib')
        clf_svc = load('classifieur_svc_mean.joblib')
        clf_rf = load('classifieur_rf_mean.joblib')
        #gbcl = load('classifieur_gbcl_mean.joblib')
        xgb = load('classifieur_xgb_mean.joblib')
        model = load_model("classifieur_rnd_mean.h5") 
        ###########################

        ##################################
        # Recharger le scaler et l'encoder

        #importation du fichier dont la moyenne a été imputée aux valeur Na
        df_mean = pd.read_csv('W_Aus_Na_mean.csv', index_col = 0)
        #remplacement des modalités des variables 'RainTomorrow' et 'RainToday' par 1 ou 0
        df_mean['RainTomorrow']=df_mean['RainTomorrow'].replace({'No':0,'Yes':1})
        df_mean['RainToday']=df_mean['RainToday'].replace({'No':0,'Yes':1})
        #suppression de la colonne 'Date' qui n'a pas d'incidence fondamentale sur le résultat final
        df_mean.drop(['Location','year','day','State'],axis=1,inplace=True)
        # création des DataFrame features et target
        features_mean = df_mean.drop('RainTomorrow',axis=1)
        target_mean = df_mean['RainTomorrow']
        # création d'une liste des variable catégorielle
        l = []
        for i in features_mean.columns:
            if features_mean.dtypes[i]=='O':
                l.append(i)
        # encoder les variables catégorielle avec la classe LebelEncoder
        la = LabelEncoder()
        for i in l:
            features_mean[i] = la.fit_transform(features_mean[i])
        # Centrer et réduire les variables numériques
        scaler = StandardScaler()
        features_mean = scaler.fit_transform(features_mean)
        ##########################

        ##########################
        # création d'une series pandas correspondant à l'observation entrée par l'utilisateur
        X_new = pd.DataFrame(
                    data = np.array([Date, Location, MinTemp, MaxTemp, Rainfall, Evaporation, Sunshine, WindGustDir, WindGustSpeed, WindDir9am, WindDir3pm, WindSpeed9am, 
                                    WindSpeed3pm, Humidity9am, Humidity3pm, Pressure9am, Pressure3pm, Cloud9am, Cloud3pm, Temp9am, Temp3pm, RainToday]),
                    columns = ['Date', 'Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustDir', 'WindGustSpeed', 
                               'WindDir9am', 'WindDir3pm', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 
                               'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 'RainToday']
                            )

        # post traitement des données
        def get_day(date):
            splits = date.split('-')    
            day = splits[2]
            return day

        def get_month(date):
            return date.split('-')[1]

        def get_year(date):
            return date.split('-')[0]   

        # Application des fonctions
        day = X_new['Date'].apply(get_day)
        month = X_new['Date'].apply(get_month)
        year = X_new['Date'].apply(get_year)

        X_new['day'] = days
        X_new['month'] = months
        X_new['year'] = years
        X_new['year_month']= years+"-"+months
        # changement de type de donnée des colonnes month, day, et year
        X_new = X_new.astype({'year':'int64','month':'int64','day':'int64'})

        X_new['RainToday']= X_new['RainToday'].replace({'No':0,'Yes':1})
        X_new.drop(['Location','year','day'],axis=1,inplace=True)

        # encoder les variables de X_new

        for i in l:
            X_new[i] = la.transform(X_new[i])
        # Centrer et réduire les variables numériques de X_new
        X_new = scaler.transform(X_new)
        ##########################

        ##########################
        # Prediction de la probabilité de pluie pour X_new
        if modele == "Regression logistique":
            y_new_prob = lr.predict_proba(X_new)
        elif modele =="KNN":
            y_new_prob = clf_knn.predict_proba(X_new)
        elif modele =="SVM":
            y_new_prob = clf_svc.predict_proba(X_new)
        elif modele =="RandomForest":
            y_new_prob = clf_rf.predict_proba(X_new)
        elif modele =="GradientBoostingClassifier":
            y_new_prob = gbcl.predict_proba(X_new)
        elif modele =="XGboost":
            y_new_prob = xgb.predict_proba(X_new)
        elif modele =="Réseau de neurones dense":
            y_new_prob = model.predict_proba(X_new)
        else:
            st.write("Vous n'avez sélectionné aucun modèle")

        st.write('La probabilité de pluie pour demain est de :', y_new_prob*100)
        if y_new_prob > 0.5:
            st.write('Vous devriez penser à prendre votre parapluie demain !')

            image = Image.open('umbrella.png')
            # Pour centrer l'image
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(' ')
            with col2:
                st.image(image, caption='Your favorite umbrella')
            with col3:
                st.write(' ')
        else:
            st.write('Il ne semble pas nécessaire de prendre votre parapluie demain !')
