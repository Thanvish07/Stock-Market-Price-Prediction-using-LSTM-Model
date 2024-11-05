from numpy import array
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st

start = '2010-01-01'
end = '2022-12-06'

st.title('Stock Price Prediction Using LSTM')

user_input = st.text_input('Enter Stock Name')

df = data.DataReader(user_input, 'yahoo', start, end)

# Describing Data
st.subheader('Data from 01-2010 to 12-2022')
st.write(df.describe())

# Visualization-1
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
st.pyplot(fig)


# Visualization-2
st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)


# Visualization-3
st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)


# Splitting Data into Training and Testing

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

scaler = MinMaxScaler(feature_range=(0, 1))

data_training_array = scaler.fit_transform(data_training)


# Loading the Model
model = load_model('keras_model.h5')

# Testing Part
past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)


scaler = scaler.scale_
scaler_factor = 1/scaler
y_predicted = y_predicted * scaler_factor
y_test = y_test * scaler_factor


# Final Graph


st.subheader('Predictions vs Original Price')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'blue', label='Original Price')
plt.plot(y_predicted, 'red', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)


x_input = input_data[977:].reshape(1, -1)
temp_input = list(x_input)
temp_input = temp_input[0].tolist()

# demonstrate prediction for next 10 days

lst_output = []
n_steps = 100
i = 0
while (i < 30):

    if (len(temp_input) > 100):
        # print(temp_input)
        x_input = np.array(temp_input[1:])
        print("{} day input {}".format(i, x_input))
        x_input = x_input.reshape(1, -1)
        x_input = x_input.reshape((1, n_steps, 1))
        # print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i, yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input = temp_input[1:]
        # print(temp_input)
        lst_output.extend(yhat.tolist())
        i = i+1
    else:
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i = i+1

day_new = np.arange(1, 101)
day_pred = np.arange(101, 131)

fig3 = plt.figure(figsize=(12, 6))
plt.plot(df[3156:], label='Original Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig3)

fig4 = plt.figure(figsize=(12, 6))
plt.plot(lst_output, label='Predicted Price for Next 30 days')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig4)
