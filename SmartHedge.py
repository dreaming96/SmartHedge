import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import norm, poisson, gamma
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
import joblib

class SmartHedge():
    def __init__(self, M, r, K, ttm, sigma, call_delta, put_delta, S0, S1, C0, C1, delta_diff, u):
        #features
        self.M = M
        self.r = r
        self.K = K
        self.call_delta = call_delta
        self.put_delta = put_delta
        self.ttm = ttm
        self.sigma = sigma
        self.S0 = S0
        self.S1 = S1
        self.C0 = C0
        self.C1 = C1
        self.delta_diff = delta_diff
        self.u = u

    def LSTM(self):
        self.C1.index, self.S1.index = self.u.index, self.S0.index
        trainX, trainY = [], []
        n_future = 1
        n_past = 1
        # df = np.array([self.M, self.S0, self.S1, self.C0, self.C1, pd.DataFrame(np.dot(self.ttm,self.sigma)), self.delta_diff[1:]]).T
        # df = np.array([self.M, pd.DataFrame(self.ttm), self.C1]).T

        #PRICING
        # df = np.array([self.M, pd.DataFrame(self.ttm), pd.DataFrame(self.C0)]).T

        #GREEKS
        df = np.array([self.S0, self.S1, pd.DataFrame(self.K * np.ones_like(self.S0)), pd.DataFrame(np.dot(self.ttm, self.sigma)), pd.DataFrame((self.C1 - self.C0) / (self.S1 - self.S0))]).T
        scaler = MinMaxScaler(feature_range=(0, 1))
        for idx, col in enumerate(df.T):
            scaler = scaler.fit(col)
            joblib.dump(scaler, "scaler" + str(idx) + ".save")
            normalized_col = scaler.transform(col)
            df[:,:,idx] = normalized_col.T


        for i in range(n_past, len(df[0]) - n_future + 1):
            trainX.append(df[0][i - n_past:i, 0:4])
            trainY.append(df[0][i - n_past:i, 4])

        trainX, trainY = np.array(trainX), np.array(trainY)

        #LSTM PRICING
        # model = tf.keras.models.Sequential()
        # model.add(tf.keras.layers.LSTM(64, activation='ReLU', input_shape=(trainX.shape[1], trainX.shape[2]), recurrent_dropout=0.2, return_sequences=True))
        # model.add(tf.keras.layers.LSTM(32, activation='ReLU', recurrent_dropout=0.2, return_sequences=False))
        # model.add(tf.keras.layers.Dense(trainY.shape[1]))
        # model.compile(optimizer='adam', loss='mse')
        # model.summary()

        #LSTM GREEKS
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.LSTM(64, activation='sigmoid', input_shape=(trainX.shape[1], trainX.shape[2]), recurrent_dropout=0.2, return_sequences=True))
        model.add(tf.keras.layers.LSTM(32, activation='sigmoid', recurrent_dropout=0.2, return_sequences=False))
        model.add(tf.keras.layers.Dense(trainY.shape[1]))
        model.compile(optimizer='adam', loss='mse')
        model.summary()

        history = model.fit(trainX, trainY, epochs=100, batch_size=64, validation_split=0.1, verbose=1)

        SmartHedge.plots(range(len(history.history["val_loss"])), pd.DataFrame(history.history["val_loss"]), pd.DataFrame(history.history["loss"]), "Validation loss", "Training loss")
        forecast = model.predict(trainX)

        for idx, col in enumerate(df.T):
            scaler = joblib.load("scaler" + str(idx) + ".save")
            df[:,:,idx] = scaler.inverse_transform(col.T)

        prediction_unnormalized = scaler.inverse_transform(forecast)

        # SmartHedge.plots(self.ttm[0:-1], self.C0[0:-1], prediction_unnormalized, "Black-Scholes price", "LSTM price")
        SmartHedge.plots(self.ttm[0:-1], self.call_delta[0:-1], prediction_unnormalized, "Black-Scholes delta", "LSTM delta")

        delta = SmartHedge.sensitivities(self, prediction_unnormalized)
        SmartHedge.plots(self.ttm[0:-1], self.call_delta[0:-1], delta, "BS analytical delta", "LSTM analytical delta")
        LSTM_hedged_account = SmartHedge.predicted_hedged_acc(self, prediction_unnormalized)

        print("123")

    def sensitivities(self, prediction):
        LSTM_delta = np.ones_like(prediction)
        for idx, price in enumerate(prediction):
            # d1 = (np.log(self.S0[0:-1][0][idx] / self.K) + (self.r + (self.sigma ** 2)/2) * self.ttm[0:-1][idx]) / self.sigma * np.sqrt(self.ttm[0:-1][idx])
            # LSTM_delta[idx] = norm.cdf(d1)
            try:
                LSTM_delta[idx] = (prediction[idx + 1] - prediction[idx]) / (self.S0[0][idx + 1] - self.S0[0][idx])
            except:
                pass
        return LSTM_delta

    def predicted_hedged_acc(self, model_delta):
        underlying_pnl = self.S0.diff()
        underlying_pnl = underlying_pnl.fillna(0)
        option_pnl = self.C0.diff()
        option_pnl = option_pnl.fillna(0)

        hedge = - model_delta * self.S0



        return None

    @staticmethod
    def plots(x, y1, y2, label1, label2):
        plt.plot(x, y1, label=label1)
        plt.plot(x, y2, label=label2)
        plt.legend(loc='best')
        plt.show()
        plt.clf()
        return None

