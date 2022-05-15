import datetime as dt
import math
import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import norm, poisson, gamma

class numerical_methods():
    def __init__(self, paths):
        self.S0 = 100
        self.r = 0.1
        self.mu = 0.05
        self.sigma = 0.2
        self.T = 1
        self.N = 252
        self.I = 500
        self.paths = paths

    def random_numbers(self):
        random.seed(1)
        dt = float(self.T)/self.N
        self.paths = np.zeros((self.N + 1, self.I), np.float64)
        self.paths[0] = self.S0
        for t in range(1, self.N+1):
            rand = np.random.standard_normal(self.I)
            self.paths[t] = self.paths[t - 1] * np.exp((self.r - 0.5 * self.sigma ** 2) * dt + self.sigma * np.sqrt(dt) * rand)
        self.paths = pd.DataFrame(self.paths)
        self.plot()
        return self.paths

    def OU_MC(self):
        return None

    def plot(self):
        plt.title("Price paths")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.plot(self.paths, linewidth=0.5)
        #plt.savefig('MC_sims.png')
        return None


class BS_pricer():
    def __init__(self, S, call_delta, call_gamma, call_vega, call_theta, put_delta, put_gamma, put_vega, put_theta):
        self.S = S
        self.K = 100
        self.r = 0.1
        self.ttm = 1
        self.sigma = 0.2
        self.d1 = (np.log(self.S / self.K) + (self.r + (self.sigma ** 2)/2) * self.ttm) / self.sigma * np.sqrt(self.ttm)
        self.d2 = self.d1 - self.sigma * np.sqrt(self.ttm)
        self.call_delta = call_delta
        self.call_gamma = call_gamma
        self.call_vega = call_vega
        self.call_theta = call_theta
        self.put_delta = put_delta
        self.put_gamma = put_gamma
        self.put_vega = put_vega
        self.put_theta = put_theta

    def call_price(self):
        price = self.S * norm.cdf(self.d1) - self.K * np.exp(-self.r * self.ttm) * norm.cdf(self.d2)
        return price

    def put_price(self):
        price = self.K * np.exp(-self.r * self.ttm) - self.S + (self.S * norm.cdf(self.d1) - self.K * np.exp(-self.r * self.ttm) * norm.cdf(self.d2))
        return price

    def greeks(self, type):
        if type.upper() == "CALL":
            self.call_delta.append(norm.cdf(self.d1))
            self.call_gamma.append(norm.pdf(self.d1) / (self.S * self.sigma * np.sqrt(self.ttm)))
            self.call_vega.append(self.S * norm.pdf(self.d1) * np.sqrt(self.ttm))
            self.call_theta.append(- ((self.S * norm.pdf(self.d1) * self.sigma) / 2 * np.sqrt(self.ttm)) - self.r * self.K * np.exp(-self.r * self.ttm) * norm.cdf(self.d2))
        else:
            self.put_delta.append(norm.cdf(self.d1) - 1)
            self.put_gamma.append(norm.pdf(self.d1) / (self.S * self.sigma * np.sqrt(self.ttm)))
            self.put_vega.append(self.S * norm.pdf(self.d1) * np.sqrt(self.ttm))
            self.put_theta.append(- ((self.S * norm.pdf(self.d1) * self.sigma) / 2 * np.sqrt(self.ttm)) + self.r * self.K * np.exp(-self.r * self.ttm) * norm.cdf(-self.d1))


class hedged_account():
    def __init__(self, delta, gamma, spot_price, option_position):
        self.delta = delta
        self.gamma = gamma
        self.spot_price = spot_price
        self.option_position = option_position

    @staticmethod
    def pnl(underlying, option):
        underlying_pnl = underlying.diff()
        underlying_pnl = underlying_pnl.fillna(0)
        option_pnl = option.diff()
        option_pnl = option_pnl.fillna(0)
        return underlying_pnl, option_pnl


    def delta_hedging(self):
        hedge = - self.delta * self.spot_price
        return hedge

    def rebalance(self):
        return None