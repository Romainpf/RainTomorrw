
import streamlit as st

import pandas as pd
import numpy as np
from PIL import Image


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

st.markdown("<h1 style='text-align: center; color: black;'>Comment on fait cela ?</h1>", unsafe_allow_html=True)
st.markdown('''On dispose pour cette prédiction d'une base de données labelisée : 
* On entraine un modèle de machine learning de classification
* On réalise la prédiction.
''')

st.write("-------------------------------------------------")

st.markdown("<h1 style='text-align: center; color: black;'>De quelles données dispose-t-on ?</h1>", unsafe_allow_html=True)

