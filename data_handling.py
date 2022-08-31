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
        self.I = 50
        self.paths = paths

    def random_numbers(self):
        random.seed()
        dt = float(self.T)/self.N
        self.paths = np.zeros((self.N + 1, self.I), np.float64)
        self.paths[0] = self.S0
        for t in range(1, self.N+1):
            rand = np.random.standard_normal(self.I)
            self.paths[t] = self.paths[t - 1] * np.exp((self.r - 0.5 * self.sigma ** 2) * dt + self.sigma * np.sqrt(dt) * rand)
        self.paths = pd.DataFrame(self.paths)
        # self.plot()
        return self.paths

    def OU_MC(self):
        return None

    def plot(self):
        plt.title("Price paths")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.plot(self.paths, linewidth=0.5)
        #plt.savefig('MC_sims.png')
        plt.clf()
        return None


class BS_pricer():
    def __init__(self, S, call_delta, call_gamma, call_vega, call_theta, put_delta, put_gamma, put_vega, put_theta, ttm):
        self.S = S
        self.K = 100
        self.r = 0.1
        self.ttm = ttm
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
    #buy 1 call and 1 put at t0
    #adjust hedge daily
    #measure hedged pnl
    def __init__(self, delta, gamma, spot_price, option_price, option_position):
        self.delta = delta
        self.gamma = gamma
        self.spot_price = spot_price
        self.option_price = option_price
        self.option_position = option_position
        self.ttm = [n/len(self.delta) for n in range(1,len(self.delta))]
        self.ttm.reverse()
        self.r = 0.1

    @staticmethod
    def pnl(underlying, option):
        underlying_pnl = underlying.diff()
        underlying_pnl = underlying_pnl.fillna(0)
        option_pnl = option.diff()
        option_pnl = option_pnl.fillna(0)
        return underlying_pnl, option_pnl

    def bank(self, hedge1, hedge2):
        account = []
        account.append(np.exp(-self.ttm*self.r)*(hedge1 + hedge2))
        return None

    def delta_hedging(self):
        hedge = - self.delta * self.spot_price
        return hedge

    def rebalance(self, pnl, hedge):
        delta_diff = self.delta.diff()
        delta_diff = delta_diff.fillna(0)
        portfolio = pnl[0] * 0
        portfolio_unhedged = pnl[0] * 0
        for idx, i in enumerate(self.spot_price):
            for idx2, j in enumerate(self.spot_price[idx]):
                if idx2 == 0:
                    portfolio[idx][0] = self.option_price[idx][0]
                    portfolio_unhedged[idx][0] = self.option_price[idx][0]
                else:
                    print("calculating hedged pnl on day "+ str(idx) + ", path " + str(idx2))
                    portfolio[idx][idx2] = self.option_price[idx][0] + delta_diff[idx][idx2] * pnl[0][idx][idx2]
                    portfolio_unhedged[idx][idx2] = self.option_price[idx][0] + pnl[0][idx][idx2]
        hedge_pnl = portfolio.diff()
        hedge_pnl = hedge_pnl.fillna(0)
        plt.hist(hedge_pnl)
        plt.clf()
        return portfolio, delta_diff, portfolio_unhedged

