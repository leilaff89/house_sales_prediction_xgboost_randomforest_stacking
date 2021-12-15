'''
This application aims to explore 3 different types of regression model sets, namely:
Boosting, Bagging and Stacking. The predictions made are for home prices in the city of Perth - Australia
'''

# __________________________________________________________________________________________________
# libraries
import warnings

#import joblib

#import numpy as np
#import pandas as pd

import streamlit as st
from PIL import Image

import requests
#import pickle
import pandas as pd
import numpy as np
#import xgboost

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')

# __________________________________________________________________________________________________

# Loading and manipulating datasets

url1 = "https://raw.githubusercontent.com/leilaff89/house_sales_prediction/main/streamlit_app/house_prices_ml_features.csv"
url2 = "https://raw.githubusercontent.com/leilaff89/house_sales_prediction/main/streamlit_app/house_prices_no_null.csv"
url3 = "https://www.travelsafe-abroad.com/wp-content/uploads/timthumb.jpeg"

df1 = pd.read_csv(url2, encoding='iso-8859-1')
df2 = pd.read_csv(url1, encoding='iso-8859-1')

df1 = df1.rename(columns={"LATITUDE": "latitude", "LONGITUDE": "longitude"})
df1.NEAREST_SCH_DIST = df1.NEAREST_SCH_DIST.round(2)
df1.GARAGE = pd.to_numeric(df1.GARAGE, downcast='integer')

# Split
X = df2.drop(columns=['PRICE'],axis =1).values
y = df2['PRICE'].values
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGB fit model
xgb = XGBRegressor(colsample_bytree=0.6, n_estimators=160, random_state = 42)
xgb.fit(X_train, y_train)

# __________________________________________________________________________________________________

# App title

st.title("House Price Sales Prediction: Perth City")
st.write("### Explore different ensemble approaches for Regression")

# Header image

image = Image.open(requests.get(url3, stream=True).raw)
#image = Image.open(url3)
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

st.title('House Price Predictor')
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

# Loading the models
#scaler = pickle.load(open("scaler.pkl", 'rb' ))
#stack = pickle.load(open("stack.sav", 'rb' ))
#xgb = pickle.load(open("xgb.pkl", 'rb' ))
#scaler = joblib.load(open("scaler.pkl", 'rb'))

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
            #y1 = stack.predict(X)
            #y1 = np.exp(y1)
            #y1 = y1.tolist()[0]
            #y1 = f' $ {round(y1, 2):,}'
            st.title("Unavailable")

        elif ensemble=="XGBoost (Boosting)":
            # Predict
            y2 = xgb.predict(X)
            y2 = np.exp(y2)
            y2 = y2.tolist()[0]
            y2 = f' $ {round(y2, 2):,}'
            st.title(y2)

        elif ensemble=="Random Forest (Bagging)":
            # Predict
            #y3 = rfc.predict(X)
            #y3 = np.exp(y3)
            #y3 = y3.tolist()[0]
            #y3 = f' $ {round(y3, 2):,}'
            st.title("Unavailable")

# About the authors:
st.sidebar.write('')
st.sidebar.markdown('---')

st.sidebar.write('Developed by Leila FF and Alex R')
st.sidebar.write('This is a final project of Data Science, Data Analytics and Machine Learning Bootcamp at Infnet Institute.')
st.sidebar.write('Linkedin:')
st.sidebar.write('https://www.linkedin.com/in/leila-fabiola-ferreira-31675163/')
st.sidebar.write('https://www.linkedin.com/in/alexrodriguesdeoliveira/')

