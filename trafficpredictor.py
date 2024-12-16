import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf
from pickle import dump, load

def make_data(i, road_name):
    data = pd.read_csv(f"Seoul_2024/Seoul{i:02d}.csv")
    data['일자'] = pd.to_datetime(data['일자'], format='%Y%m%d')
    data.fillna(0, inplace=True)

    for i in range(24):
        if data[f"{i}시"].dtype!="float":
            data[f"{i}시"] = data[f"{i}시"].map(lambda x: float(str(x).replace(",", "").split()[0]))

    data_weather = pd.read_csv("Weather.csv")
    data_weather['날짜'] = pd.to_datetime(data_weather["날짜"])
    data_weather.fillna(0, inplace=True)

    data1 = data[data['지점명']==road_name]
    data_np = np.zeros((data1.shape[0]*24//2, 5))
    
    day_to_num = {"일":0, "월":1, "화":2, "수":3, "목":4, "금":5, "토":6}
    
    for i in range(data1.shape[0]//2):
        for j in range(24):
            # print(data1[data1["일자"]==data1["일자"].iloc[i]][f"{j}시"].iloc[0])
            data_np[i*24+j,0] = data1[f"{j}시"].iloc[i]
            data_np[i*24+j,1] = data1[data1["일자"]==data1["일자"].iloc[i]][f"{j}시"].iloc[1]
            data_np[i*24+j,2] = j
            data_np[i*24+j,3] = day_to_num[data1["요일"].iloc[i]]
            data_np[i*24+j,4] = data_weather[data_weather['날짜']==data1['일자'].iloc[i]]["강수량(mm)"].iloc[0]
    
    return data_np

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def train_model(road_name, seq_length=12):
    data_np = []
    for i in range(9):
        data_np.append(make_data((i+1), road_name))
    data_np = np.concatenate(data_np)
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data_np)
    
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]

    X_train, y_train = create_sequences(train_data, seq_length)
    X_test, y_test = create_sequences(test_data, seq_length)
    
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(seq_length, 5)),
        Dense(5)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
    
    model.save(f'models/model_{road_name}.keras')
    dump(scaler, open(f'models/scaler_{road_name}.pkl', 'wb'))
    return model, scaler

def train_val_model(road_name, seq_length=12):
    data_np = []
    for i in range(9):
        data_np.append(make_data((i+1), road_name))
    data_np = np.concatenate(data_np)
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data_np)
    
    train_size = int(len(scaled_data) * 0.8 * 0.8)
    val_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    validation_data = scaled_data[train_size:val_size]
    test_data = scaled_data[val_size:]

    X_train, y_train = create_sequences(train_data, seq_length)
    X_val, y_val = create_sequences(validation_data, seq_length)
    X_test, y_test = create_sequences(test_data, seq_length)
    
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(seq_length, 5)),
        Dense(5)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    hist = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32, verbose=0)
    
    return model, scaler, hist

def seoul_model_predict(given_data, road_name, seq_length=12, forecast_period=30):
    model = tf.keras.models.load_model(f'models/model_{road_name}.keras')
    scaler = load(open(f'models/scaler_{road_name}.pkl', 'rb'))
    
    scaled_data = scaler.fit_transform(given_data)
    
    forecast = []
    X_data, y_data = create_sequences(scaled_data, seq_length)
    # Use the last sequence from the test data to make predictions
    last_sequence = X_data[-1]
    next_temp = scaler.inverse_transform(last_sequence[-1:])
    time_cnt = int(next_temp[0,2])
    day_cnt = int(next_temp[0,3])
    
    for _ in range(forecast_period):
        time_cnt = time_cnt + 1
        if time_cnt==24:
            time_cnt = 0
            day_cnt = (day_cnt + 1) % 7
        
        # Reshape the sequence to match the input shape of the model
        current_sequence = last_sequence.reshape(1, seq_length, 5)
        # Predict the next value
        next_prediction = model.predict(current_sequence, verbose=0)
        next_temp = scaler.inverse_transform(next_prediction)
        next_temp[0,2] = time_cnt
        next_temp[0,3] = day_cnt
        next_temp[0,4] = 0
        next_prediction = scaler.transform(next_temp)
        # Append the prediction to the forecast list
        forecast.append(next_prediction)
        # Update the last sequence by removing the first element and appending the predicted value
        last_sequence = np.concatenate((last_sequence[1:], next_prediction))
    
    # Inverse transform the forecasted values
    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 5))

    return forecast