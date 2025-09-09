import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import streamlit as st

# Predefined stock tickers (Popular Companies)
STOCK_TICKERS = {
    "Apple Inc. (AAPL)": "AAPL",
    "Microsoft Corp. (MSFT)": "MSFT",
    "Amazon.com Inc. (AMZN)": "AMZN",
    "Google (Alphabet) (GOOGL)": "GOOGL",
    "Tesla Inc. (TSLA)": "TSLA",
    "NVIDIA Corp. (NVDA)": "NVDA",
    "Meta Platforms (META)": "META",
    "Netflix Inc. (NFLX)": "NFLX",
    "Bitcoin USD (BTC-USD)": "BTC-USD",
    "Ethereum USD (ETH-USD)": "ETH-USD",
}

def run_stock_trend_prediction():
    start = '2015-01-01'
    end = '2024-12-31'

    st.title('Stock Trend Prediction')

    # Dropdown for Ticker Selection
    selected_stock = st.selectbox("Select a Stock", list(STOCK_TICKERS.keys()))
    user_input = STOCK_TICKERS[selected_stock]

    df = yf.download(user_input, start=start, end=end)

    if df.empty:
        st.error("No data found for this ticker.")
        return

    st.subheader(f'Data for {selected_stock} ({user_input}) from 2015-2024')
    st.write(df.describe())

    # Plot Closing Price vs Time
    st.write("### Closing Price vs Time Chart")
    fig = plt.figure(figsize=(12, 6))
    plt.plot(df.Close, label='Closing Price')
    plt.title('Closing Price vs Time')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    st.pyplot(fig)

    # Moving Averages
    ma100 = df.Close.rolling(100).mean()
    ma200 = df.Close.rolling(200).mean()

    st.write("### Closing Price with 100-Day and 200-Day Moving Averages")
    fig = plt.figure(figsize=(12, 6))
    plt.plot(ma100, label='100-Day Moving Average', color='orange')
    plt.plot(ma200, label='200-Day Moving Average', color='green')
    plt.plot(df.Close, label='Closing Price', color='blue')
    plt.title('Closing Price vs Time with Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig)

    # Data Splitting for Model Prediction
    data_training = df['Close'][:int(len(df) * 0.70)]
    data_testing = df['Close'][int(len(df) * 0.70):]

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))

    data_training_array = scaler.fit_transform(data_training.values.reshape(-1, 1))

    model = load_model('keras_model_3.h5')

    past_100_days = data_training[-100:]
    final_df = pd.concat([past_100_days, data_testing])
    input_data = scaler.transform(final_df.values.reshape(-1, 1))

    x_test = []
    y_test = []

    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100:i])
        y_test.append(input_data[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)

    y_predicted = model.predict(x_test)

    # Rescale the predictions
    scale_factor = 1 / scaler.scale_[0]
    y_predicted = y_predicted * scale_factor
    y_test = y_test * scale_factor

    test_dates = df.index[int(len(df) * 0.70):]

    # Predictions vs Actual
    st.write("### Predictions vs Actual Prices")
    fig2 = plt.figure(figsize=(12, 6))
    plt.plot(test_dates, y_test, 'b', label='Original Price')
    plt.plot(test_dates, y_predicted, 'r', label='Predicted Price')
    plt.title(f'Predictions vs Actual Prices for {selected_stock} ({user_input})')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig2)
