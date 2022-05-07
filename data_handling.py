import datetime as dt
import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import norm, poisson, gamma

class numerical_methods():
    def __init__(self, paths):
        self.S0 = 100
        self.r = 0.01
        self.mu = 0.05
        self.sigma = 0.2
        self.T = 1
        self.N = 252*10
        self.I = 1000
        self.paths = paths

    def MC(self):
        dt = float(self.T)/self.N
        self.paths = np.zeros((self.N + 1, self.I), np.float64)
        self.paths[0] = self.S0
        for t in range(1, self.N+1):
            rand = np.random.standard_normal(self.I)
            self.paths[t] = self.paths[t - 1] * np.exp((self.r - 0.5 * self.sigma ** 2) * dt + self.sigma * np.sqrt(dt) * rand)
        self.plot()
        return self.paths

    def plot(self):
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.plot(self.paths, linewidth=0.5)
        #plt.savefig('MC_sims.png')
        return None

class BS_pricer():
    def __init__(self):
        self.S = S
        self.K = K
        self.r = r
        self.T = T
        self.sigma = sigma
        self.d1 = (np.log(self.S / self.K) + (self.r + self.sigma ** 2) * T) / self.sigma * np.sqrt(self.T)
        self.d2 = self.d1 - self.sigma * np.sqrt(self.T)

    def call_price(self):
        price = self.S * norm.cdf(self.d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2)
        return price

    def put_price(self):
        price = self.K * np.exp(-self.r * self.T) - self.S + (self.S * norm.cdf(self.d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2))
        return price

    def greeks(self, price, type):
        call_delta = norm.cdf(self.d1)
        call_gamma = norm.pdf(self.d1) / (self.S * self.sigma * np.sqrt(self.T))
        call_vega = self.S * norm.pdf(self.d1) * np.sqrt(self.T)
        call_theta = - ((self.S * norm.pdf(self.d1) * self.sigma) / 2 * np.sqrt(self.T)) - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2)
        put_delta = norm.cdf(self.d1) - 1
        put_gamma = norm.pdf(self.d1) / (self.S * self.sigma * np.sqrt(self.T))
        put_vega = self.S * norm.pdf(self.d1) * np.sqrt(self.T)
        put_theta = - ((self.S * norm.pdf(self.d1) * self.sigma) / 2 * np.sqrt(self.T)) + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d1)
        if type.upper() == "CALL":
            return call_delta, call_gamma, call_vega, call_theta
        else:
            return put_delta, put_gamma, put_vega, put_theta
