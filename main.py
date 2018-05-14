# From https://medium.com/@haydar_ai/learning-data-science-day-9-linear-regression-on-boston-housing-dataset-cd62a80775ef

from matplotlib import rcParams # special matplotlib argument for improved plots
import matplotlib.pyplot as plt
import numpy as np
import pandas
import seaborn as sns
import scipy.stats as stats
from sklearn.datasets import load_boston
import statsmodels.api as sm

sns.set_style("whitegrid")
sns.set_context("poster")

boston = load_boston()

if __name__ == "__main__":
    data = pandas.DataFrame(boston.data)
    data.columns = boston.feature_names
    data['PRICE'] = boston.target
    print(data.head())
    print(data)
