
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt


model=joblib.load('solarproject.pkl')

#create streamlit 
st.title('solar power generation prediction')
st.write('enter the features for solar power prediction')


# Load the model
@st.cache_data
def load_model():
    return joblib.load('solarproject.pkl')

model = load_model()
#features input
    

with st.sidebar:
    st.header('input header')
    distance_to_solar_noon=st.number_input('distance-to-solar-noon',min_value=0.00000,max_value=10.0000,value=0.0000)
    temperature=st.number_input('temperature',min_value=0,max_value=100,value=0)
    wind_direction=st.number_input('wind-direction',min_value=0,max_value=100,value=0)
    wind_speed=st.number_input('wind-speed',min_value=0.0,max_value=10.0,value=0.0)
    sky_cover=st.number_input('sky-cover',min_value=0,max_value=4,value=0)
    visibility=st.number_input('visibility',min_value=0.0,max_value=10.0,value=0.0)
    humidity	=st.number_input('humidity',min_value=0,max_value=100,value=14)
    average_wind_speed=st.number_input('average-wind-speed-(period)',min_value=0.0,max_value=10.0,value=0.0)
    average_pressure=st.number_input('average-pressure-(period)',min_value=0.00,max_value=35.00,value=0.0)
#power_generated=st.number_input('power-generated',min_value=0,max_value=40000,value=0)

input_features=np.array([[distance_to_solar_noon, temperature, wind_direction, wind_speed,
       sky_cover, visibility, humidity, average_wind_speed,
       average_pressure]])

if st.button('Predict'):
    features = np.array(input_features)
    prediction = model.predict(features)
    st.success(f'xgboost prediction : {prediction[0]:.2f}')
    
#uploading fil csv
upload_file=st.file_uploader("upload the csv file",type=["csv"])
if upload_file is not None:
    data=pd.read_csv(upload_file)
    st.write('upload_file')
    st.write(data)
    predict_file=model.predict(data)
    data['prediction']=predict_file
    st.write('prediction')
    st.write(data)
    #download the predicted file
    st.download_button (
        
        label="download_prediction",
        data=data.to_csv(index="False").encode('utf-8'),
        file_name='predicted.csv',
        mime='csv/text'
    )




if st.button("Show Predictions Plot"):
    plt.figure(figsize=(10, 6))
    plt.plot(predict_file, label='Predictions')
    plt.xlabel('Samples')
    plt.ylabel('Solar Energy Output')
    plt.title('Predicted Solar Energy Output')
    plt.legend()
    st.pyplot(plt)
        
    
