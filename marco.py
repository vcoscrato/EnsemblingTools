import numpy as np
import pickle

with open('error.pkl', 'rb') as f:
    error = pickle.load(f)

nns = error[0]
x_val = error[1]
y_val = error[2]

x_error = x_val[67].reshape(1, -1)
print('x problematico:', x_error)

for i, est in enumerate(nns.estimators):
    print('predicoes base', i, ':', est.predict(x_error))

print('predicao nns', nns.predict(x_error))
print('thetas nns', nns.get_weights(x_error))

