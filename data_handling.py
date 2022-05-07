import datetime as dt
import math
import numpy as np
import pandas as pd
from matplotlib import pyplot
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
        return self.paths

    def plot(self):

        return None

if __name__ == "__main__":
    instance = numerical_methods([])
    numerical_methods.MC(instance)
    print(123)

