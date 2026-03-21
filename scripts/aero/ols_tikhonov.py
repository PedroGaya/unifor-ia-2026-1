import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('data/aerogerador.dat')
X = data[:,0]
y = data[:,1]

X_ones = np.column_stack([np.ones_like(X), X])
lam = 1 # lambda from {0, 0.25, 0.5, 0.75, 1, 10000}
beta_hat = np.linalg.inv(X_ones.T @ X_ones + lam * np.eye(2)) @ X_ones.T @ y

x_line = np.linspace(X.min(), X.max(), 200)
y_line = beta_hat[0] + beta_hat[1] * x_line # p + 1 = 2

fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(X, y)
ax.plot(x_line, y_line, color='red')

ax.set_xlabel('Wind speed (m/s)')
ax.set_ylabel('Power (kW)')
ax.set_title('Power curve - Regularized OLS', fontweight='bold')

ax.grid(True, linestyle='--', alpha=0.4)

def validate(X, y, lam, rounds=500, test_size=0.2):
    n = len(y)
    n_test = int(n * test_size)
    mse_scores = np.zeros(rounds)
    r2_scores  = np.zeros(rounds)
 
    for i in range(rounds):
        idx = np.random.permutation(n)
        idx_test, idx_train = idx[:n_test], idx[n_test:]
 
        X_train, y_train = X[idx_train], y[idx_train]
        X_test,  y_test  = X[idx_test],  y[idx_test]
 
        b = np.linalg.inv(X_train.T @ X_train + lam * np.eye(2)) @ X_train.T @ y_train
        y_pred = X_test @ b
 
        mse_scores[i] = np.mean((y_test - y_pred) ** 2)
        r2_scores[i]  = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
 
    for name, scores in [('MSE', mse_scores), ('R2', r2_scores)]:
        print(f'{name} mean={scores.mean():.4f} std={scores.std():.4f}'
              f' min={scores.min():.4f} max={scores.max():.4f}')

validate(X_ones, y, lam)

# plt.tight_layout()
# plt.savefig(f'out/ols_tikhonov_{lam}.png')
# plt.show()