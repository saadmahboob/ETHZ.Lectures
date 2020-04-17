#Linear regression on noisy data

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import scipy

# generate noisy data with sigma = noise and fit with linear regression
def run_noisy_fit(noise):
    # generate random x values
    np.random.seed(42)
    x = np.sort(np.random.rand(200)).reshape(-1, 1)

    # list containing random noise vectors
    epsilon = [
        ["Gaussian", np.random.normal(0, noise, len(x))], # symmetric distribution with sigma = noise
        ["Cauchy", scipy.stats.cauchy.rvs(loc=0, scale=noise, size=len(x))], # symmetric distribution, heavier tails
        #["Gamma", np.random.gamma(0.5, noise, len(x))], # asymmetric distribution with heavy tails
    ]

    # iterate through all noise vectors
    for i,(name, eps) in enumerate(epsilon):
        # construct y measurements as x + noise
        y = x + eps.reshape(-1, 1)

        # fit linear regression model
        reg = LinearRegression().fit(x, y)

        # make predictions on training data
        y_pred = reg.predict(x)

        # plot training data
        plt.scatter(x, y, c="C" + str(i))

        # plot predicted response on training points
        plt.plot(x,
                 y_pred,
                 label="{}: w1={}, SE/N={}".format(name,reg.coef_[0][0], mean_squared_error(x, y_pred)),
                 c="C" + str(i)
        )
    plt.legend()
    plt.show()


run_noisy_fit(0.05)


# map vector x of m data points to (m x 2*n + 1) matrix of m feature vectors
def map_to_features(x, n):
    return np.hstack([
        np.ones(len(x)).reshape(-1, 1),
        np.sin(np.outer(x, np.arange(n) + 1)),
        np.cos(np.outer(x, np.arange(n) + 1)),
    ])



# generate noisy data with sigma = noise, with n features per data point
def periodic_fit(noise, n):
    # generate random x values
    np.random.seed(42)
    x = 8 * np.pi * np.sort(np.random.rand(180)).reshape(-1, 1)

    # list containing random noise vectors
    epsilon = [
        ("Gaussian", np.random.normal(0, noise, len(x))),
        #("Gaussian", np.random.normal(0, noise * 2, len(x))),
        #("Gaussian", np.random.normal(0, noise * 10, len(x))),
    ]

    # iterate through all noise vectors
    for i,(name, eps) in enumerate(epsilon):
        # generate measurements
        y = np.sin(x) + eps.reshape(-1, 1)

        # map inputs to feature vectors
        X = map_to_features(x, n)

        #fit linear regression model
        reg = LinearRegression().fit(X, y)

        # make predictions on training data
        y_pred = reg.predict(X)
        plt.scatter(x, y, c="C" + str(i))
        plt.plot(x,
                 y_pred,
                 label="{}: w1={}, SE/N={}".format(name,reg.coef_[0][0], mean_squared_error(x, y_pred)),
                 c="C" + str(i)
        )

    plt.show()

periodic_fit(0.005, 50)
