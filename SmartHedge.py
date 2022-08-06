import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#inputs:
# S: stock price
# K: strike price
# r: risk-free rate
# T: maturity
# h: hint (?) [option delta?, BS hedge?]

class SmartHedge():
    def __init__(self, S, K, r, T, call_delta, put_delta):
        self.S = S
        self.K = K
        self.r = r
        self.T = T
        self.call_delta = call_delta
        self.put_delta = put_delta

    def LSTM(self):
        train = np.array(self.S[:120])
        test = np.array(self.S[121:])
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.LSTM(64, activation='relu', input_shape=(train.shape[0], train.shape[1]), return_sequences=True))
        model.add(tf.keras.layers.LSTM(32, activation='relu', return_sequences=False))
        model.add(tf.keras.layers.Dropout(0,2))
        model.add(tf.keras.layers.Dense(32))
        model.compile(optimizer='adam', loss='mse')
        model.summary()
        return None
