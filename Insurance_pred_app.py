import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
with open('XGBoost_model.pkl','rb') as file:
    model_xgb = pickle.load(file)
st.title('Car Insurane prediction')
st.write("this is a app to check the Car Insurane prediction using car features")
st.subheader("Enter Car details to check the result")


policy_tenure = st.number_input("policy tenure", min_value=0.002735272840513, max_value=1.39664107699389)
age_of_car = st.number_input("age of Car", min_value=0.0, max_value=0.245)
age_of_policyholder = st.number_input("age of policyholder", min_value=0.288461538461538, max_value=0.822115384615385)
population_density = st.number_input("population density", min_value=290, max_value=58340)
is_adjustable_steering = st.selectbox("is adjustable steering select 1 for Yes and 0 for NO",['0','1'])
cylinder = st.selectbox("No. of Cylinder in a Engine",['3','4'])
width = st.number_input("Width of a car in cm", min_value=1475, max_value=1811)
is_front_fog_lights = st.selectbox("is front fog lights is availble select 1 for Yes and 0 for NO",['0','1'])
is_brake_assist = st.selectbox("is brake assist is availble select 1 for Yes and 0 for NO",['0','1'])
is_driver_seat_height_adjustable = st.selectbox("is seat height adjustable select 1 for Yes and 0 for NO",['0','1'])


Input_Data = pd.DataFrame({
    "policy tenure":[policy_tenure],
    "age of car" : [age_of_car],
    "age of policyholder": [age_of_policyholder],
    "population density":[population_density],
    "is adjustable steering select 1 for Yes and 0 for NO":[is_adjustable_steering],
    "No. of Cylinder in a Engine":[cylinder],
    "Width of a car in cm":[width],
    "is front fog lights is availble select 1 for Yes and 0 for NO":[is_front_fog_lights],
    "is brake assist is availble select 1 for Yes and 0 for NO":[is_brake_assist],
    "is seat height adjustable select 1 for Yes and 0 for NO":[is_driver_seat_height_adjustable]
    })


Expected_order = ['policy_tenure', 'age_of_car', 'age_of_policyholder',
       'population_density', 'is_adjustable_steering', 'cylinder', 'width',
       'is_front_fog_lights', 'is_brake_assist',
       'is_driver_seat_height_adjustable']

Input_data = Input_Data.reindex(columns=Expected_order)

if st.button("predict Insurance"):
    prediction = model_xgb.predict(Input_data)
    result_mapping = {0:'No Insurance',  1: 'Insurance'}
    result = result_mapping[prediction[0]]
    st.write(f'The Car Insurance prediction is : {result}')