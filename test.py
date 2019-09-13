import numpy as np
import os
import warnings
import pandas as pd
import pickle
from time import time
from nnstacking import NNS, LinearStack, NNPredict
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.exceptions import DataConversionWarning
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_predict, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib
from matplotlib import pyplot as plt
matplotlib.rcParams['text.usetex'] = True

def metrics(cv_predictions, y, t):

    output = np.empty((cv_predictions.shape[1], 5))
    for model in range(cv_predictions.shape[1]):
        preds = cv_predictions[:, model]
        index = np.isnan(preds)
        output[model, 0] = mean_squared_error(preds[~index], y[~index])
        output[model, 1] = np.std((cv_predictions[:, model]-y)**2) / (len(y)**(1/2))
        output[model, 2] = mean_absolute_error(preds[~index], y[~index])
        output[model, 3] = np.std(abs(cv_predictions[:, model]-y)) / (len(y)**(1/2))
        output[model, 4] = t[model]

    output = pd.DataFrame(output)
    output.columns = ['MSE', 'MSE_STD', 'MAE', 'MAE_STD', 'Time']
    output.index = ['LSR', 'Lasso', 'Ridge', 'Bagging', 'RF', 'GBR', 'UNNS', 'UNNSp', 'CNNS', 'CNNSp', 'Breiman', 'NNMeta', 'NN']

    return output

data = pd.read_csv('/home/vcoscrato/Datasets/blog feedback.csv')
x = data.iloc[:, range(0, data.shape[1] - 1)].values
y = data.iloc[:, -1].values
with open('fitted/blog feedback.pkl', 'rb') as f:
    cvpreds = pickle.load(f)

for index, (train, test) in enumerate(KFold(n_splits=4, shuffle=True, random_state=0).split(x)):
    print(metrics(cvpreds[0][test], y[test], cvpreds[2]))


