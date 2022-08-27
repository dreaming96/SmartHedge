import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
import joblib

#inputs:
# S: stock price
# K: strike price
# S/K = OR moneyness of option ^^^^
# r: risk-free rate
# T: maturity
# h: hint (?) [option delta?, BS hedge?]
# STATIONARITY OF INPUTS

#minimize mean squared hedging error on hedged portfolio?

class SmartHedge():
    def __init__(self, M, r, ttm, sigma, call_delta, put_delta, S0, S1, C0, C1, delta_diff):
        #features
        self.M = M
        self.r = r
        self.call_delta = call_delta
        self.put_delta = put_delta
        self.ttm = ttm
        self.sigma = sigma
        #additional
        self.S0 = S0
        self.S1 = S1
        self.C0 = C0
        #target
        self.C1 = C1
        self.delta_diff = delta_diff

    def LSTM(self):
        trainX, trainY = [], []
        n_future = 1
        n_past = 1
        data = []
        #train on one price path, test on multiple price paths
        #INCLUDE VOL * SQRT TIME - SAME FOR MC PRICING
        df = np.array([self.M, self.S0, self.S1, self.C0, self.C1, pd.DataFrame(np.dot(self.ttm,self.sigma)), self.delta_diff[1:]]).T
        scaler = MinMaxScaler(feature_range=(0, 1))
        for idx, col in enumerate(df.T):
            scaler = scaler.fit(col)
            joblib.dump(scaler, "scaler" + str(idx) + ".save")
            normalized_col = scaler.transform(col)
            df[:,:,idx] = normalized_col.T


        for i in range(n_past, len(df[0]) - n_future + 1):
            trainX.append(df[0][i - n_past:i, 0:5])
            trainY.append(df[0][i - n_past:i, 5])

        trainX, trainY = np.array(trainX), np.array(trainY)

        model = tf.keras.models.Sequential()
        # model.add(tf.keras.layers.Normalization())
        model.add(tf.keras.layers.LSTM(64, activation='ReLU', input_shape=(trainX.shape[1], trainX.shape[2]), recurrent_dropout=0.2, return_sequences=True))
        model.add(tf.keras.layers.LSTM(32, activation='ReLU', recurrent_dropout=0.2, return_sequences=False))
        model.add(tf.keras.layers.Dense(trainY.shape[1]))
        model.compile(optimizer='adam', loss='mse')
        model.summary()

        history = model.fit(trainX, trainY, epochs=100, batch_size=64, validation_split=0.1, verbose=1)

        plt.plot(history.history["loss"], label="Training loss")
        plt.plot(history.history["val_loss"], label="Validation loss")
        plt.legend()
        plt.show()
        n_future = 90
        forecast = model.predict(trainX)

        for idx, col in enumerate(df.T):
            scaler = joblib.load("scaler" + str(idx) + ".save")
            df[:,:,idx] = scaler.inverse_transform(col.T)
        forecast_unnormalized = scaler.inverse_transform(forecast)



        print("123")


        return None
