'''
This application aims to explore 3 different types of regression model sets, namely:
Boosting, Bagging and Stacking. The predictions made are for home prices in the city of Perth - Australia
'''

# __________________________________________________________________________________________________
# libraries
import warnings

import joblib

import numpy as np
import pandas as pd

import streamlit as st
from PIL import Image

warnings.filterwarnings('ignore')

# __________________________________________________________________________________________________

# Loading and manipulating datasets

df1 = pd.read_csv("house_prices_no_null.csv", encoding='iso-8859-1')
df2 = pd.read_csv("house_prices_ml_features.csv", encoding='iso-8859-1')

#new_df = df1.groupby('SUBURB', as_index=False)['PRICE'].mean()
#replace_dict = new_df.set_index('SUBURB').to_dict()['PRICE']
#df1 = df1.replace(replace_dict)
#df2.SUBURB = df2.SUBURB.round(2)
df1.NEAREST_SCH_DIST = df1.NEAREST_SCH_DIST.round(2)
df1.GARAGE = pd.to_numeric(df1.GARAGE, downcast='integer')

# __________________________________________________________________________________________________

# App title

st.title("House Price Sales Prediction: Perth City")
st.write("### Explore different ensemble approaches for Regression")

# Header image

image = Image.open('perth.jpeg')
st.image(image)
st.text("Image Source: https://www.travelsafe-abroad.com/br/australia/perth/")
# ________________________________________________________________________________________________________

"""
##  About this APP
This application aims to explore 3 different types of regression model ensembles, namely:
Boosting, Bagging and Stacking. The predictions made are for home prices in the city of Perth - Australia

"""

# ---------------------------------------------------------------------------------------------------------
'''
##  Map of Perth
The red points are sold houses in Perth and whose descriptive data was used in this project.'''
# Map
# ---------------------------------------------------------------------------------------------------------
df_map = df1[['latitude', 'longitude']]
st.map(df_map, zoom=9)

# ----------------------------------------------------------------------------------
'''### Data Summary'''

# Data summary

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.title(df1.latitude.count())
    st.text('Data size')

with col2:
    st.title(df1.SUBURB.nunique())
    st.text('Number of suburbs')

with col3:
    st.title(df1.PRICE.max())
    st.text('Most expensive house')

with col4:
    st.title(df1.FLOOR_AREA.max())
    st.text('Biggest House')

# -------------------------------------------------------------------------------------------------------
# Instructions

st.title('House Prices Predictor')
"""
Please choose an ensemble approach and house features, then click in submit buttom to get the prediction.
## Predicted Price:
"""
#---------------------------------------------------------------------------------------------------------
#Choose the ensemble

ensemble = st.sidebar.selectbox("Choose the ensemble",
                                ("XGBoost (Boosting)",
                                 "Random Forest (Bagging)",
                                 "Stacking"))

scaler = joblib.load("scaler.pkl")
stack = joblib.load("stack.pkl")
xgb = joblib.load("xgb.pkl")
rfc = joblib.load("rfc.pkl")

#------------------------------------------------------------------------------------
#Getting the user data in a form

with st.form("my_form"):
    f_SUBURB = st.sidebar.select_slider("Suburb mean prices",
                                        options=sorted(np.exp(df2.SUBURB.unique()).round()),
                                        key=1)

    f_BEDROOM = st.sidebar.select_slider("Number of Bedrooms",
                                         options=sorted(df1.BEDROOMS.unique()),
                                         key=2)

    f_BATHROOM = st.sidebar.select_slider("Number of Bathrooms",
                                          options=sorted(df1.BATHROOMS.unique()),
                                          key=3)

    f_GARAGE = st.sidebar.select_slider("Number of garages",
                                        options=sorted(df1.GARAGE.unique()),
                                        key=4)

    f_FLOOR_AREA = st.sidebar.select_slider("Floor Area Size - m2",
                                            options=sorted(df1.FLOOR_AREA.unique()),
                                            key=5)

    f_BUILD_YEAR = st.sidebar.select_slider("Build Year",
                                            options=sorted(df1.BUILD_YEAR.unique()),
                                            key=6)

    f_CBD_DIST = st.sidebar.select_slider("Downtown distance",
                                          options=sorted(df1.CBD_DIST.unique()),
                                          key=7)

    f_NEAREST_STN_DIST = st.sidebar.select_slider("Nearest Station distance",
                                                  options=sorted(df1.NEAREST_STN_DIST.unique()),
                                                  key=8)

    f_NEAREST_SCH_DIST = st.sidebar.select_slider("Nearest school distance",
                                                  options=sorted(df1.NEAREST_SCH_DIST.unique()),
                                                  key=9)

    f_NEAREST_SCH_RANK = st.sidebar.select_slider("Nearest school ranking",
                                                  options=sorted(df1.NEAREST_SCH_RANK.unique()),
                                                  key=10)

    f_YEAR_SOLD = st.sidebar.select_slider("Year sold",
                                           options=sorted(df1.YEAR_SOLD.unique()),
                                           key=11)

    f_LATITUDE = st.sidebar.select_slider("Latitude",
                                          options=sorted(df1.latitude.unique()),
                                          key=12)

    f_LONGITUDE = st.sidebar.select_slider("Longitude",
                                           options=sorted(df1.longitude.unique()),
                                           key=13)
# ----------------------------------------------------------------------------------------
    # Manipulating user data

    df_to_predict = ([[np.log(f_SUBURB),f_BEDROOM,f_BATHROOM,f_GARAGE,np.log(f_FLOOR_AREA),f_BUILD_YEAR,
                      f_CBD_DIST,np.log(f_NEAREST_STN_DIST),f_LATITUDE,f_LONGITUDE,np.log(f_NEAREST_SCH_DIST),
                      f_NEAREST_SCH_RANK,f_YEAR_SOLD]])

    X = np.asarray(df_to_predict)
    X = scaler.transform(X)

# ---------------------------------------------------------------------------------------------

    # Submit and predict

    submitted = st.form_submit_button("Submit")
    if submitted:
        if ensemble=="Stacking" :
            # Predict
            y1 = stack.predict(X)
            y1 = np.exp(y1)
            y1 = y1.tolist()[0]
            y1 = f' $ {round(y1, 2):,}'
            st.title(y1)

        elif ensemble=="XGBoost (Boosting)":
            # Predict
            y2 = xgb.predict(X)
            y2 = np.exp(y2)
            y2 = y2.tolist()[0]
            y2 = f' $ {round(y2, 2):,}'
            st.title(y2)

        elif ensemble=="Random Forest (Bagging)":
            # Predict
            y3 = rfc.predict(X)
            y3 = np.exp(y3)
            y3 = y3.tolist()[0]
            y3 = f' $ {round(y3, 2):,}'
            st.title(y3)

# About the authors:
st.sidebar.write('')
st.sidebar.markdown('---')

st.sidebar.write('Developed by Leila FF and Alex R')
st.sidebar.write('This is a final project of Data Science, Data Analytics and Machine Learning Bootcamp at Infnet Institute.')
st.sidebar.write('Linkedin:')
st.sidebar.write('https://www.linkedin.com/in/leila-fabiola-ferreira-31675163/')
st.sidebar.write('https://www.linkedin.com/in/alexrodriguesdeoliveira/')