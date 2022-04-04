# RainTomorrow
Introduction
Le dataset weatherAUS.csv contient des données météorologiques issues des observations de nombreuses stations météos réparties sur l’ensemble du territoire Australien. Le dataset téléchargeable sur le site internet Kaggle, compile environ 10 ans d’observations quotidiennes fournies par le bureau of meteorology du gouvernement australien (http://www.bom.gov.au/climate/data/), qui met à disposition de nombreuses données en libre accès.
 
Ce dataset sera donc utilisé dans le cadre du projet de groupe de la formation Data Scientist dispensée par DataScientest et dont les objectifs seront les suivants :
>* Exploration du Dataset en s’appuyant sur des visualisations pertinentes qui permettront de mieux appréhender la relation des variables entre elles et les phénomènes météorologiques australiens
>* Choisir et paramétrer un modèle de machine learning performant afin de prédire la variable « RainTomorrow », mais prédire également :  
    - La température moyenne issue des variables « Temp3pm », « Temp9pam », « MaxTemp », « MinTemp »  
    - Et enfin la variable « WindGustDir »  
>- Réaliser une prédiction à long terme en utilisant des techniques mathématiques de séries temporelles ou des réseaux de neurones récurrents.  

Le dataset est composé de 23 variables dont les définitions sont les suivantes :

**Date**  date des mesures au format aaaa-mm-jj   
**Location** = 49 localités enregistrées  
**MinTemp** = température minimale en C° relevée dans les 24h qui précèdent 9h00  
**MaxTemp** = température maximale en C° relevée dans les 24h suivant 9h00  
**Rainfall** = précipitations en mm mesurées dans les 24h qui suivent 9h00 le matin  
**Evaporation** = évaporation en mm mesurée dans les 24h qui suivent 9h00 le matin  
**Sunshine** = Nombre d'heures d'ensoleillement mesurées dans les 24h qui suivent 00h00  
**WindGustDir** = Direction de la plus forte rafale de vent mesurée dans les 24h qui suivent 00h00 - 16 modalités différentes sont enregistrées  
**WindGustSpeed** = Vitesse en km/h de la plus forte rafale de vent mesurée dans les 24h qui suivent 00h00  
**WindDir9am** = Direction moyenne du vent durant les 10min précédent 9h - 16 modalités différentes sont enregistrées  
**WindDir3pm** = Direction moyenne du vent durant les 10min précédent 15h - 16 modalités différentes sont enregistrées  
**WindSpeed9am** = Vitesse en km/h moyenne du vent durant les 10min précédent 9h  
**WindSpeed3pm** = Vitesse en km/h moyenne du vent durant les 10min précédent 15h  
**Humidity9am** = taux d'humidité en % à 9h  
**Humidity3pm** = taux d'humidité en % à 15h  
**Pressure9am** = pression atmosphérique à 9h enregistrée en hectopascals et ramenée au niveau moyen de la mer  
**Pressure3pm** = pression atmosphérique à 15h enregistrée en hectopascals et ramenée au niveau moyen de la mer  
**Cloud9am** = Fraction du ciel obscurcie par les nuages à 9 heures du matin  
**Cloud3pm** = Fraction du ciel obscurcie par les nuages à 15h  
**Temp9am** = Température à 9h  
**Temp3pm** = Température à 15h  
**RainToday** = "Yes" si au moins moins mm de pluie dans la journée, sinon non  
**RainTomorrow** = "Yes" si au moins moins mm de pluie dans la journée, sinon non  
