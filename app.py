import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import requests
import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu

st.set_page_config(layout="centered")

st.markdown(
   f"""
   <style>
   .stApp {{
        background-image: url("https://static.vecteezy.com/system/resources/previews/001/372/960/original/abstract-flat-green-and-blue-background-free-vector.jpg");
        background-attachment: fixed;
        background-size: cover
   }}
   </style>
   """,
   unsafe_allow_html=True)
hide_menu_style = """
            <style>
            #MainMenu {visibility : hidden; }
            footer {visibility : hidden; }
            </style>
            """
st.markdown(hide_menu_style, unsafe_allow_html=True)
   
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Home","About Us", "About Project"],
        menu_icon= "cast",
        icons = ["house", "person-lines-fill","book"],
    )
# def load_lottieurl(url):
#     r = requests.get(url)
#     if r.status_code != 200:
#         return None
#     return r.json()

# lottie_coding = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_f2jo61ci.json")

if selected == "Home":
    
    start = '2012-01-01'
    end = '2022-12-01'

    st.markdown("<h1 style='text-align: center;font-family: baskerville, serif'><u>Stock Price Predictor</u></h1>", unsafe_allow_html=True)


    user_input = st.text_input('Enter Stock Ticker', 'TCS.NS')
    df = data.DataReader(user_input, 'stooq', start, end)

    st.markdown("<h4 style='text-align: center'>Data From 2012-2022</h1>", unsafe_allow_html=True)
    st.write(df.describe())

    st.markdown("<h4 style='text-align: center'>Closing Price VS Time Chart</h1>", unsafe_allow_html=True)
    fig = plt.figure(figsize=(12,6))
    plt.plot(df.Close)
    st.pyplot(fig)

    st.markdown("<h4 style='text-align: center'>Closing Price VS Time Chart with 100MA</h1>", unsafe_allow_html=True)
    ma100 = df.Close.rolling(100).mean()
    fig = plt.figure(figsize=(12,6))
    plt.plot(ma100, 'r')
    plt.plot(df.Close)
    st.pyplot(fig)

    st.markdown("<h4 style='text-align: center'>Closing Price VS Time Chart with 100MA & 200MA</h1>", unsafe_allow_html=True)
    ma100 = df.Close.rolling(100).mean()
    ma200 = df.Close.rolling(200).mean()
    fig = plt.figure(figsize=(12,6))
    plt.plot(ma100, 'r')
    plt.plot(ma200, 'g')
    plt.plot(df.Close)
    st.pyplot(fig)

    data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
    data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70) : int(len(df))])

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0,1))

    data_training_array = scaler.fit_transform(data_training)

    model = load_model('keras_model.h5')

    past_100_days = data_training.tail(100)
    final_df = past_100_days.append(data_testing, ignore_index = True)
    input_data = scaler.fit_transform(final_df)

    x_test = []
    y_test = []

    for i in range(100,input_data.shape[0]):
        x_test.append(input_data[i-100:i])
        y_test.append(input_data[i,0])

    x_test, y_test = np.array(x_test), np.array(y_test)
    y_predicted = model.predict(x_test)
    scaler = scaler.scale_
    scale_factor = 1/scaler[0]
    y_predicted = y_predicted * scale_factor
    y_test = y_test * scale_factor

    st.markdown("<h4 style='text-align: center'>Prediction VS Orignal</h1>", unsafe_allow_html=True)
    fig2 = plt.figure(figsize = (12,6))
    plt.plot(y_test, 'b', label = 'Orignal Price')
    plt.plot(y_predicted, 'r', label = 'Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig2)

if selected == "About Us":
    st.markdown("<h1 style='text-align: center; font-family: baskerville, serif'><u>About Us</u></h1>", unsafe_allow_html=True)
    st.write("""<p style = 'font-family: couier, monospace'>
    The Stock Price Predictor Webapp has been made by Students Of PSIT College Of Engineering College Students.
    Below Here is their name and Linkedin Profile URL:-</p>""", unsafe_allow_html= True)
    st.write("(1) ANUBHAV TRIPATHI-----CS-III-AI-----2003481520004----[Linkedin](https://www.linkedin.com/in/anubhav-tripathi-656959210/)")
    st.write("(2) KARTIKEY PANDEY-----CS-III-AIML-----2003481530011----[Linkedin](https://www.linkedin.com/in/kartikey-pandey-44b720206/)")
    st.write("(3) SHIVANSH TRIPATHI-----CS-III-AIML-----2003481530022----[Linkedin](https://www.linkedin.com/in/shivansh-tripathi-a4118b238/)")
    st.write("(4) SIDDHARTH SRIVASTVA-----CS-III-AIML-----2003481530024----[Linkedin](https://www.linkedin.com/in/siddharth-srivastava-960091223/)")

if selected == "About Project":
    
    st.markdown("<h1 style='text-align: center; font-family: baskerville, serif'><u>About Project</u></h1>", unsafe_allow_html=True)
    with st.container():
        right_coloumn = st.columns(1)
        st.write("""<h4 style = 'text-align : center; font-family: rockwell, monotype foundry'>Greetings And Welcome To Stock Price Predictor.</h4>""", unsafe_allow_html=True)
        st.write("""<p style = 'font-family: couier, monospace'>The Stock Price Preictor is a mini project which is made with the help of Machine Learning and Web Devlopement
                    and is used to predict the values of stock trends.<br>
                    Stock Price Predictor helps you discover the future value of company stock and other financial assets traded on an exchange.
                    The entire idea of predicting stock prices is to gain significant profits. 
                    Predicting how the stock market will perform is a hard task to do.<br><br>
                    There are other factors involved in the prediction, such as physical and psychological factors, rational and irrational behavior, and so on. 
                    All these factors combine to make share prices dynamic and volatile. 
                    This makes it very difficult to predict stock prices with high accuracy.<br><br>
                    For our exercise, we will be looking at technical analysis solely and focusing on the Simple MA and Exponential MA techniques to predict stock prices. 
                    Additionally, we will utilize LSTM (Long Short-Term Memory), a deep learning framework for time-series, to build a predictive model and compare its performance against our technical analysis. 
                    As stated in the disclaimer, stock trading strategy is not in the scope of this article. 
                    Stock Price Predictor will be using trading/investment terms only to help you better understand the analysis, but this is not financial advice.</p> """,unsafe_allow_html=True)