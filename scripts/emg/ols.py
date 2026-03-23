import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('data/EMGsDataset.csv', delimiter=',')
X = data[:2].T
y = data[2].astype(int)
C = y.max()
N = len(y)

def fit_predict(X_tr, y_tr, X_te):
    Y_tr = np.zeros((len(y_tr), C))
    Y_tr[np.arange(len(y_tr)), y_tr - 1] = 1
    W = np.linalg.pinv(X_tr) @ Y_tr
    return (X_te @ W).argmax(axis=1) + 1

ROUNDS     = 500
TEST_RATIO = 0.2
rng = np.random.default_rng(42)

def validate(fit_predict_fn):
    acc = np.zeros(ROUNDS)
    for r in range(ROUNDS):
        idx    = rng.permutation(N)
        split  = int(N * (1 - TEST_RATIO))
        tr, te = idx[:split], idx[split:]
        y_hat  = fit_predict_fn(X[tr], y[tr], X[te])
        acc[r] = (y_hat == y[te]).mean()
    return acc

acc = validate(fit_predict)
print(f'{"OLS":<22}  {acc.mean():>7.2%}  {acc.std():>7.2%}  {acc.min():>7.2%}  {acc.max():>7.2%}')