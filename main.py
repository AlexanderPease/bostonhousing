# From https://medium.com/@haydar_ai/learning-data-science-day-9-linear-regression-on-boston-housing-dataset-cd62a80775ef

from matplotlib import rcParams # special matplotlib argument for improved plots
import matplotlib.pyplot as plt
import numpy as np
import pandas
import seaborn as sns
import scipy.stats as stats
# import sklearn
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

sns.set_style("whitegrid")
sns.set_context("poster")


class BostonData(object):
    def __init__(self):
        boston = load_boston()
        data = pandas.DataFrame(boston.data)
        data.columns = boston.feature_names
        data['PRICE'] = boston.target

        self.data = data
        self.features = self.data.drop('PRICE', axis=1)
        self.target = self.data['PRICE']

    def dump(self):
        print(data.describe())

    def split_data(self):
        X_train, X_test, Y_train, Y_test = train_test_split(
            self.features,
            self.target,
            test_size = 0.33,
            random_state = 5
        )

        # Train the model
        lm = LinearRegression()
        lm.fit(X_train, Y_train)

        # These are predicted target values
        Y_pred = lm.predict(X_test)

        # Visualize predicted vs actual Y test values...
        # Couldn't get visualizations to run
        # plt.scatter(Y_test, Y_pred)
        # plt.xlabel("Prices: $Y_i$")
        # plt.ylabel("Predicted prices: $\hat{Y}_i$")
        # plt.title("Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")

        # ...which is mathematically capture in Mean Squared Error
        mse = mean_squared_error(Y_test, Y_pred)
        print('Mean Squared Error: {}'.format(mse))


if __name__ == "__main__":
    data = BostonData()
    data.split_data()
