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

warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)

def empty(*args, **kwargs):
    return np.empty(*args, **kwargs) + np.nan

def cvpredict(x, y, base_est, NN_layers):

    # Top level cross-validation
    splitter = KFold(n_splits=4, shuffle=True, random_state=0)
    cv_predictions = empty((len(x), len(base_est)+7))
    thetas = empty((4, len(x), len(base_est)))
    t = [0]*(len(base_est)+7)
    for index, (train, test) in enumerate(splitter.split(x)):

        print('Fold:', index)

        # Inner data splitting
        x_train, y_train = x[train], y[train]
        x_strain, x_val, y_strain, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=0)

        # UNNS
        print('UNNS...')
        best_mse = np.infty
        t0 = time()
        predictions_splits = []
        kwargs = dict(
		verbose=0,
                nn_weight_decay=0.0,
                es=True,
                es_give_up_after_nepochs=50,
                num_layers=-1,
                hidden_size=100,
                estimators=base_est,
                ensemble_method="UNNS",
                ensemble_addition=False,
                es_splitter_random_state=0,
		)
        for layers in NN_layers:
            print('Current layers:', layers)
            kwargs['num_layers'] = layers
            nns = NNS(**kwargs).fit(x_strain, y_strain.reshape(-1, 1), None)
            error = mean_squared_error(nns.predict(x_val), y_val)
            if error < best_mse:
                best_mse = error
                best_model = layers

	    # append predictions for reuse
            predictions_splits.append(nns.predictions)

        kwargs['num_layers'] = best_model
        print("Best number of layers:", best_model)
        print("Fitting model with full data")
        nns = NNS(**kwargs).fit(x_train, y_train.reshape(-1, 1), None)
        best_model = nns
        best_mse = mean_squared_error(nns.predict(x_val), y_val)
        predictions_full = nns.predictions

        t[len(base_est)] += time() - t0
        cv_predictions[test, len(base_est)] = best_model.predict(x[test]).flatten()
        thetas[0, test, :] = best_model.get_weights(x[test])[0]

        # Fit base estimators
        print('Base estimators...')
        for i, est in enumerate(base_est):
            t0 = time()
	    # estimators were fitted already by the above function
            # est.fit(x[train], y[train])
            cv_predictions[test, i] = est.predict(x[test])
            t[i] += time() - t0

        # UNNS + phi
        print('UNNS + phi...')
        best_mse = np.infty
        t0 = time()
        predictions_splits = []
        kwargs = dict(
		verbose=0,
                nn_weight_decay=0.0,
                es=True,
                es_give_up_after_nepochs=50,
                num_layers=-1,
                hidden_size=100,
                estimators=base_est,
                ensemble_method="UNNS",
                ensemble_addition=True,
                es_splitter_random_state=0,
		)
        for lidx, layers in enumerate(NN_layers):
            print('Current layers:', layers)
            kwargs['num_layers'] = layers
            nns = NNS(**kwargs).fit(x_strain, y_strain.reshape(-1, 1), predictions_splits[lidx])
            error = mean_squared_error(nns.predict(x_val), y_val)
            if error < best_mse:
                best_mse = error
                best_model = layers

        kwargs['num_layers'] = best_model
        print("Best number of layers:", best_model)
        print("Fitting model with full data")
        nns = NNS(**kwargs).fit(x_train, y_train.reshape(-1, 1), predictions_full)
        best_model = nns
        best_mse = mean_squared_error(nns.predict(x_val), y_val)

        t[len(base_est)+1] += time() - t0
        cv_predictions[test, len(base_est)+1] = best_model.predict(x[test]).flatten()
        thetas[1, test, :] = best_model.get_weights(x[test])[0]

        # CNNS
        print('CNNS...')
        best_mse = np.infty
        t0 = time()
        predictions_splits = []
        kwargs = dict(
		verbose=0,
                nn_weight_decay=0.0,
                es=True,
                es_give_up_after_nepochs=50,
                num_layers=-1,
                hidden_size=100,
                estimators=base_est,
                ensemble_method="CNNS",
                ensemble_addition=False,
                es_splitter_random_state=0,
		)
        for lidx, layers in enumerate(NN_layers):
            print('Current layers:', layers)
            kwargs['num_layers'] = layers
            nns = NNS(**kwargs).fit(x_strain, y_strain.reshape(-1, 1), predictions_splits[lidx])
            error = mean_squared_error(nns.predict(x_val), y_val)
            if error < best_mse:
                best_mse = error
                best_model = layers

        kwargs['num_layers'] = best_model
        print("Best number of layers:", best_model)
        print("Fitting model with full data")
        nns = NNS(**kwargs).fit(x_train, y_train.reshape(-1, 1), predictions_full)
        best_model = nns
        best_mse = mean_squared_error(nns.predict(x_val), y_val)

        t[len(base_est)+2] += time() - t0
        cv_predictions[test, len(base_est)+2] = best_model.predict(x[test]).flatten()
        thetas[2, test, :] = best_model.get_weights(x[test])[0]

        # CNNS + phi
        print('CNNS + phi...')
        best_mse = np.infty
        t0 = time()
        predictions_splits = []
        kwargs = dict(
		verbose=0,
                nn_weight_decay=0.0,
                es=True,
                es_give_up_after_nepochs=50,
                num_layers=-1,
                hidden_size=100,
                estimators=base_est,
                ensemble_method="CNNS",
                ensemble_addition=True,
                es_splitter_random_state=0,
		)
        for lidx, layers in enumerate(NN_layers):
            print('Current layers:', layers)
            kwargs['num_layers'] = layers
            nns = NNS(**kwargs).fit(x_strain, y_strain.reshape(-1, 1), predictions_splits[lidx])
            error = mean_squared_error(nns.predict(x_val), y_val)
            if error < best_mse:
                best_mse = error
                best_model = layers

        kwargs['num_layers'] = best_model
        print("Best number of layers:", best_model)
        print("Fitting model with full data")
        nns = NNS(**kwargs).fit(x_train, y_train.reshape(-1, 1), predictions_full)
        best_model = nns
        best_mse = mean_squared_error(nns.predict(x_val), y_val)

        t[len(base_est)+3] += time() - t0
        cv_predictions[test, len(base_est)+3] = best_model.predict(x[test]).flatten()
        thetas[3, test, :] = best_model.get_weights(x[test])[0]

        # Direct NN
        print('Direct NN...')
        best_mse = np.infty
        t0 = time()
        predictions_splits = []
        kwargs = dict(
            verbose=0,
            es_give_up_after_nepochs=50,
            hidden_size=100,
            num_layers=-1,
		)
        for layers in NN_layers:
            print('Current layers:', layers)
            kwargs['num_layers'] = layers
            nnpredict = NNPredict(**kwargs).fit(x_strain, y_strain.reshape(-1, 1))
            error = mean_squared_error(nnpredict.predict(x_val), y_val)
            if error < best_mse:
                best_mse = error
                best_model = layers

        kwargs['num_layers'] = best_model
        print("Best number of layers:", best_model)
        print("Fitting model with full data")
        nnpredict = NNPredict(**kwargs).fit(x_train, y_train)
        best_model = nnpredict
        best_mse = mean_squared_error(nnpredict.predict(x_val), y_val)

        t[len(base_est)+6] += time() - t0
        cv_predictions[test, len(base_est)+6] = best_model.predict(x[test]).flatten()

        # Breiman's stacking
        print('Breiman\'s...')
        t0 = time()
        if len(train) > 50000:
            linear = LinearStack(estimators=base_est, verbose=0).fit(x[train[:50000]], y[train[:50000]], g[:50000])
        else:
            linear = LinearStack(estimators=base_est, verbose=0).fit(x[train], y[train], g)

        t[len(base_est)+4] += time() - t0
        cv_predictions[test, len(base_est)+4] = linear.predict(x[test])

        # NN-Meta
        print('NN-Meta...')
        g_train, g_val, y_train, y_val = train_test_split(g, y[train], test_size=0.1, random_state=0)

        best_mse = np.infty
        t0 = time()
        for layers in NN_layers:
            print('Current layers:', layers)
            model = Sequential()
            for j in range(layers):
                model.add(Dense(100, input_shape=(g.shape[1],), activation='elu'))
                model.add(Dropout(0.5))
            model.add(Dense(1, activation='linear'))
            model.compile(loss="mse", optimizer='adam')
            callbacks = [EarlyStopping(monitor='loss', patience=50, restore_best_weights=True)]
            model.fit(g_train, y_train, epochs=1000, batch_size=128, verbose=0, callbacks=callbacks)
            error = mean_squared_error(model.predict(g_val), y_val)
            if error < best_mse:
                best_mse = error
                best_model = model

        g_test = empty((len(test), len(base_est)))
        for i, est in enumerate(base_est):
            g_test[:, i] = est.predict(x[test])

        t[len(base_est)+5] += time() - t0
        cv_predictions[test, len(base_est)+5] = best_model.predict(g_test).flatten()

    return cv_predictions, thetas, t

def metrics(cv_predictions, y, t):

    output = empty((cv_predictions.shape[1], 5))
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

def weights_plot(thetas, file_name, results):

    which = np.argmin(results.iloc[6:10, 1].values)
    w = pd.DataFrame(thetas[which])
    w.columns = ['LSR', 'Lasso', 'Ridge', 'Bagging', 'RF', 'GBR']

    f, ax = plt.subplots()
    ax.set_ylabel('Weights', fontsize=18)
    w.boxplot(fontsize='large')
    f.savefig(file_name, bbox_inches='tight')

    return 'Success!'


if __name__ == '__main__':

    # Superconductivity
    data = pd.read_csv('datasets/superconductivity.csv')
    x = data.iloc[:, range(0, data.shape[1] - 1)].values
    y = data.iloc[:, -1].values

    frname = 'fitted/superconductivity.pkl'
    try:
        with open(frname, 'rb') as f:
            cvpreds = pickle.load(f)
    except (IOError, EOFError):
        cvpreds = cvpredict(x, y, NN_layers=[1, 3, 10],
              base_est=[
                  LinearRegression(),
                  LassoCV(),
                  RidgeCV(),
                  GridSearchCV(BaggingRegressor(n_jobs=-1), {
                      'n_estimators': (5, 10, 20, 50)
                  }),
                  RandomForestRegressor(n_jobs=-1),
                  GridSearchCV(GradientBoostingRegressor(), {
                      'learning_rate': (0.01, 0.1, 0.2),
                      'n_estimators': (50, 100, 200)
                  }, n_jobs=-1)
              ])
        with open(frname, 'wb') as f:
            pickle.dump(cvpreds, f, pickle.HIGHEST_PROTOCOL)

    results = metrics(cvpreds[0], y, cvpreds[2])
    results.to_csv('results/superconductivity.csv', index_label = 'Model')
    weights_plot(cvpreds[1], 'img/superconductivity.pdf', results)

    # Blog
    data = pd.read_csv('datasets/blog feedback.csv')
    x = data.iloc[:, range(0, data.shape[1] - 1)].values
    y = data.iloc[:, -1].values
    """
    cvpreds = cvpredict(x, y, NN_layers=[1, 3, 10],
          base_est=[
              LinearRegression(),
              LassoCV(),
              RidgeCV(),
              GridSearchCV(BaggingRegressor(n_jobs=-1), {'n_estimators': (5, 10, 20, 50)}),
              RandomForestRegressor(n_jobs=-1),
              GridSearchCV(GradientBoostingRegressor(), {'learning_rate': (0.01, 0.1, 0.2), 'n_estimators': (50, 100, 200)}, n_jobs=-1)
          ])
    with open('fitted/blog feedback.pkl', 'wb') as f:
        pickle.dump(cvpreds, f, pickle.HIGHEST_PROTOCOL)
    """
    with open('fitted/blog feedback.pkl', 'rb') as f:
        cvpreds = pickle.load(f)

    results = metrics(cvpreds[0], y, cvpreds[2])
    results.to_csv('results/blog feedback.csv', index_label = 'Model')
    weights_plot(cvpreds[1], 'img/blog feedback.pdf', results)

    # GPU
    data = pd.read_csv('datasets/GPU kernel performance.csv')
    x = data.iloc[:, :13].values
    y = data.iloc[:, 14:].apply(np.mean, axis=1).values

    cvpreds = cvpredict(x, y, NN_layers=[1, 3, 10],
          base_est=[
              LinearRegression(),
              LassoCV(),
              RidgeCV(),
              GridSearchCV(BaggingRegressor(n_jobs=-1), {'n_estimators': (5, 10, 20, 50)}),
              RandomForestRegressor(n_jobs=-1),
              GridSearchCV(GradientBoostingRegressor(), {'learning_rate': (0.01, 0.1, 0.2), 'n_estimators': (50, 100, 200)}, n_jobs=-1)
          ])
    with open('fitted/GPU.pkl', 'wb') as f:
        pickle.dump(cvpreds, f, pickle.HIGHEST_PROTOCOL)

    results = metrics(cvpreds[0], y, cvpreds[2])
    results.to_csv('results/GPU.csv', index_label = 'Model')
    weights_plot(cvpreds[1], 'img/GPU.pdf', results)

    # Year
    data = pd.read_table('datasets/music year.txt', sep=',')
    x = data.iloc[:, range(1, data.shape[1])].values
    y = data.iloc[:, 0].values

    cvpreds = cvpredict(x, y, NN_layers=[1, 3, 10],
          base_est=[
              LinearRegression(),
              LassoCV(),
              RidgeCV(),
              GridSearchCV(BaggingRegressor(n_jobs=-1), {'n_estimators': (5, 10, 20, 50)}),
              RandomForestRegressor(n_jobs=-1),
              GridSearchCV(GradientBoostingRegressor(), {'learning_rate': (0.01, 0.1, 0.2), 'n_estimators': (50, 100, 200)}, n_jobs=-1)
          ])
    with open('fitted/year.pkl', 'wb') as f:
        pickle.dump(cvpreds, f, pickle.HIGHEST_PROTOCOL)

    results = metrics(cvpreds[0], y, cvpreds[2])
    results.to_csv('results/year.csv', index_label = 'Model')
    weights_plot(cvpreds[1], 'img/year.pdf', results)
