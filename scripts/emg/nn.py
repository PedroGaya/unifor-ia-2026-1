import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

data = np.genfromtxt('data/EMGsDataset.csv', delimiter=',')
X = data[:2].T
y = data[2].astype(int)
classes = np.unique(y)
N, P = X.shape

rounds = 10
ratio = 0.2
rng = np.random.default_rng()

variants = {
    'MLP (32,16)':      (32, 16),
    'MLP (64,32,16)':   (64, 32, 16),
    'MLP (8,)':         (8,),
}

results = {name: np.zeros(rounds) for name in variants}

for r in range(rounds):
    idx   = rng.permutation(N)
    split = int(N * (1 - ratio))
    tr, te = idx[:split], idx[split:]

    scaler  = StandardScaler().fit(X[tr])
    X_tr_s  = scaler.transform(X[tr])
    X_te_s  = scaler.transform(X[te])

    for name, layers in variants.items():
        model = MLPClassifier(
            hidden_layer_sizes=layers,
            activation='relu',
            max_iter=500,
            random_state=r,
        )
        model.fit(X_tr_s, y[tr])
        results[name][r] = (model.predict(X_te_s) == y[te]).mean()

print(f'{"Variant":<22}  {"Mean":>7}  {"Std":>7}  {"Min":>7}  {"Max":>7}')
for name, acc in results.items():
    print(f'{name:<22}  {acc.mean():>7.2%}  {acc.std():>7.2%}  {acc.min():>7.2%}  {acc.max():>7.2%}')