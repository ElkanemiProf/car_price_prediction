# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 22:48:48 2023

@author: Admin
"""
import pandas as pd
import numpy as np
import streamlit as st
import re
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px

st.write("Nigerian Car Price Prediction App")
st.write("-------")

# loading the saved model
loaded_model = pickle.load(
    open('C:/Users/Admin/Desktop/data science/MACHINE LEARNING/CAR PRICE PREDICTION/car_pred_model.sav', 'rb'))
# load the dataset
naija_df = pd.read_csv('cleaned_car.csv')
naija_df.rename(columns={'Engine Size': 'Engine_Size'}, inplace=True)
naija_df.rename(columns={'Year of manufacture': 'Year_of_manufacture'}, inplace=True)
naija_df.replace(',', '', regex=True, inplace=True)
naija_df.drop(['Unnamed: 0', 'Make'], axis=1, inplace=True)

print(naija_df.columns)
print(naija_df.head())

st.write(naija_df)
#
#  Visualization
chart_select = st.sidebar.selectbox(label='Select the type of chart'
                                    , options=['Scatterplots', 'Lineplots', 'Histogram', 'Boxplot'])

numeric_columns = list(naija_df.select_dtypes(['float', 'int']).columns)

if chart_select == 'Scatterplots':
    st.sidebar.subheader('Scatter plot settings')
    try:
        x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
        y_values = st.sidebar.selectbox('Y axis', options=numeric_columns)
        plot = px.scatter(data_frame=naija_df, x=x_values, y=y_values)
        st.write(plot)
    except Exception as e:
        print(e)
if chart_select == 'Histogram':
    st.sidebar.subheader('Histogram settings')
    try:
        x_values = st.sidebar.selectbox('X axis', options=numeric_columns)

        plot = px.histogram(data_frame=naija_df, x=x_values)
        st.write(plot)
    except Exception as e:
        print(e)
if chart_select == 'Lineplots':
    st.sidebar.subheader('Lineplots settings')
    try:
        x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
        y_values = st.sidebar.selectbox('Y axis', options=numeric_columns)
        plot = px.line(data_frame=naija_df, x=x_values, y=y_values)
        st.write(plot)
    except Exception as e:
        print(e)
if chart_select == 'Boxplot':
    st.sidebar.subheader('Boxplot settings')
    try:
        x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
        y_values = st.sidebar.selectbox('Y axis', options=numeric_columns)
        plot = px.box(data_frame=naija_df, x=x_values)
        st.write(plot)
    except Exception as e:
        print(e)
# sidebar
#  specify input parameters
st.sidebar.header('Specify input parameters')


def assign_value_to_cat(list_of_options):
    value_list = {}
    for index, j in enumerate(list_of_options):
        value_list[j] = index
    return value_list


#



#
def predict_price(Year_of_manufacture, Condition, Mileage, Engine_Size, Fuel, Transmission, Build):
    if Condition == 'Nigerian Used':
        Condition = 0
    elif Condition == 'Foreign Used':
        Condition = 1
    elif Condition == 'Brand New':
        Condition = 2


    if Transmission == 'Automatic':
        Transmission = 0
    elif Transmission == 'Manual':
        Transmission = 1
    elif Transmission == 'CVT':
        Transmission = 2
    elif Transmission == 'AMT':
        Transmission = 3


    #
    if Fuel == 'Petrol':
        Fuel = 0
    elif Fuel == 'Diesel':
        Fuel = 1
    elif Fuel == 'Hybrid':
        Fuel = 2

    if Build == 'SUV':
        Build = 0
    #
    # fuel_result = int(assign_value_to_cat(Fuels)[Fuel])

    data = np.array([Year_of_manufacture, Condition, Mileage, Engine_Size, Fuel, Transmission, Build]).reshape(1, -1)

    prediction = loaded_model.predict(data)
    return prediction


#  print specified input parameters
st.header('Specified input parameters')

Year_of_manufacture = st.text_input('manufacture_year')

Mileage = st.slider("Select a Mileage value", min_value=80, max_value=2671736, step=1)
Engine_Size = st.slider("Select an Engine Size value", min_value=3, max_value=50000, step=1)


Fuels = ["Petrol", "Diesel", "Hybrid"]
Fuel = st.selectbox("Select Fuel Type", Fuels)
Conditions = ["Nigerian Used", "Foreign Used", "Brand New"]
Condition = st.selectbox("Select Vehicle condition", Conditions)
#
Transmissions = ["Automatic", "Manual", "CVT", "AMT"]
Transmission = st.selectbox("Select Transmission type", Transmissions)

Builds = ["SUV"]
Build = st.selectbox("Select an option", Builds)

# Apply model to make prediction
if st.button("Predict"):
    prediction = predict_price(int(Year_of_manufacture), Condition, Mileage, Engine_Size, Fuel, Transmission, Build)
    st.success("The predicted price for this car is N{:,.2f}".format(prediction[0]))

