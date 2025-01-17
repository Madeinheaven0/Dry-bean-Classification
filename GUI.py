import streamlit as st
import numpy as np
from tensorflow import keras 
import joblib

#bean's class
classe = ['BARBUNYA', 'BOMBAY', 'CALI', 'DERMASON', 'HOROZ', 'SEKER', 'SIRA']

#Load the model
model = keras.models.load_model("bean_classifier.keras")

#list of features
parameters = [
            "Area", 
            "Perimeter", 
            "MajorAxisLength", 
            "MinorAxisLength", 
            "AspectRation",
            "Eccentricity",
            "ConvexArea",
            "EquivDiameter",
            "Extent",
            "Solidity",
            "roundness",
            "Compactness",
            "ShapeFactor1",
            "ShapeFactor2",
            "ShapeFactor3",
            "ShapeFactor4",
            ]
features = []

st.markdown("<h1 style='text-align:center; color:red'>BEAN'S CLASSIFIER</h1>", unsafe_allow_html=True)

"Features of the bean"
area = st.number_input("Area:", key="area", format='%.6f')
perimeter = st.number_input("Perimeter:", key="perimeter", format='%.6f')
Mal = st.number_input("MajorAxisLength:", key="MAL", format='%.6f')
mal = st.number_input("MinorAxisLength:", key="mAL", format='%.6f')
aspectration = st.number_input("Aspect ration:", key="aspectration", format='%.6f')
eccentricity = st.number_input("Eccentricity:", key="eccentricty", format='%.6f')
cxa = st.number_input("Convex area:", key="covex area", format='%.5f')
equivdiameter = st.number_input("EquivDiamater:", key="equivdiameter", format='%.6f')
extent = st.number_input("Extent", key="extent", format='%.6f')
solidity = st.number_input("Solidity:", key="solidity", format='%.6f')
roundness = st.number_input("Roundness:", key="roundness", format='%.6f')
compactness = st.number_input("Compactness:", key="compactness", format='%.6f')
SF1 = st.number_input("Shape factor 1:", key="sf1", format='%.6f')
SF2 = st.number_input("Shape factor 2:", key="sf2", format='%.6f')
SF3 = st.number_input("Shape factor 3:", key="sf3", format='%.6f')
SF4 = st.number_input("Shape factor 4:", key="sf4", format='%.6f')

features.append([area, perimeter, Mal, mal, aspectration, eccentricity, cxa, equivdiameter, extent,
                 solidity, roundness, compactness, SF1, SF2, SF3, SF4])

features = np.array(features)

det = np.argmax(model.predict(features)) #array of probability of each class

bean = classe[det]

f"Your bean is probabily the class {bean}" #The name of rhe class