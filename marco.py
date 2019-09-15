import numpy as np
import pickle
import torch

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

#Set nnx to global
def predict(self, x_pred):
    with torch.no_grad():
        self._check_dims(x_pred, np.empty((1,1)))
        for eind, estimator in enumerate(self.estimators):
            if self.verbose >= 1:
                print("Calculating prediction for estimator",
                      estimator)
            prediction = estimator.predict(x_pred)
            if len(prediction.shape) == 1:
                prediction = prediction[:, None]
            if eind == 0:
                predictions = np.empty((x_pred.shape[0],
                                        prediction.shape[1],
                                        self.est_dim))
            predictions[:, :, eind] = torch.from_numpy(prediction)
            print(predictions)
        self.neural_net.eval()
        global nnx
        nnx = _np_to_var(x_pred)
        nnpred = _np_to_var(predictions)
        output = self._ensemblize(nnx, nnpred)
        return output.data.cpu().numpy()

def _np_to_var(arr):
    arr = np.array(arr, dtype='f4')
    arr = torch.from_numpy(arr)
    return arr

nns.predict = predict
#Calculate nnx
nns.predict(nns, x_error)
#Output M^-1
output = nns.neural_net(nnx).view(-1, 6, 6)
print('M^-1:', output)
output_res = output.new(output.shape[0], 6).shape
evec = output.new_ones(6)[:, None]

#Aqui as coisas viram tudo 0
div_res = output[0]
div_res = div_res.tril()
div_res = torch.mm(div_res, div_res.t())
numerator = torch.mm(div_res, evec)
print('numerador:', numerator)
denominator = torch.mm(numerator.t(), evec)
print('denominador:', denominator)
div_res = numerator / denominator
print('thetas:', div_res)



