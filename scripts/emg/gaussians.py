import numpy as np

data = np.genfromtxt('data/EMGsDataset.csv', delimiter=',')
X = data[:2].T
y = data[2].astype(int)
classes = np.unique(y)
N, P = X.shape
C = len(classes)

def fit_predict(X_tr, y_tr, X_te, cov_fn):
    means  = {c: X_tr[y_tr == c].mean(axis=0) for c in classes}
    priors = {c: (y_tr == c).mean() for c in classes}
    covs   = cov_fn(X_tr, y_tr)
    N_te   = len(X_te)
    scores = np.zeros((N_te, C))
    for i, c in enumerate(classes):
        _, logdet = np.linalg.slogdet(covs[c])
        diff = X_te - means[c]
        scores[:, i] = (
            - 0.5 * logdet
            - 0.5 * ((diff @ np.linalg.pinv(covs[c])) * diff).sum(axis=1)
            + np.log(priors[c])
        )
    return classes[scores.argmax(axis=1)]

def cov_standard(X, y):
    return {c: np.cov(X[y == c].T) for c in classes}

def cov_equal(X, y):
    pooled = sum((y == c).sum() * np.cov(X[y == c].T) for c in classes) / len(y)
    return {c: pooled for c in classes}

def cov_aggregate(X, y):
    return {c: np.cov(X.T) for c in classes}

def cov_regularized(lam):
    def _cov(X, y):
        N = len(y)
        agg = np.cov(X.T)
        covs = {}
        for c in classes:
            N_c    = (y == c).sum()
            sigma_c  = np.cov(X[y == c].T)
            num    = (1 - lam) * N_c * sigma_c + lam * N * agg
            denom  = (1 - lam) * N_c + lam * N
            covs[c] = num / denom
        return covs
    return _cov
  
def cov_naive_bayes(X, y):
    return {c: np.diag(X[y == c].var(axis=0)) for c in classes}

rounds = 500
ratio = 0.2
rng = np.random.default_rng(42)

lambdas = [0, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

variants = {
    'Standard (QDA)':    cov_standard,
    'Equal cov (LDA)':   cov_equal,
    'Aggregate':         cov_aggregate,
    'Naive Bayes':       cov_naive_bayes,
}
for l in lambdas:
    variants[f'Regularized lambda={l}'] = cov_regularized(lam=l)

results = {name: np.zeros(rounds) for name in variants}

for r in range(rounds):
    idx  = rng.permutation(N)
    split = int(N * (1 - ratio))
    tr, te = idx[:split], idx[split:]
    for name, cov_fn in variants.items():
        y_hat = fit_predict(X[tr], y[tr], X[te], cov_fn)
        results[name][r] = (y_hat == y[te]).mean()

print(f'{"Variant":<22}  {"Mean":>7}  {"Std":>7}  {"Min":>7}  {"Max":>7}')

for name, acc in results.items():
    print(f'{name:<22}  {acc.mean():>7.2%}  {acc.std():>7.2%}  {acc.min():>7.2%}  {acc.max():>7.2%}')