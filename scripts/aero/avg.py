import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('data/aerogerador.dat')
X = data[:, 0]
y = data[:, 1]

y_mean = np.mean(y)

fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(X, y)
ax.axhline(float(y_mean), color='red')

ax.set_xlabel('Wind speed (m/s)')
ax.set_ylabel('Power (kW)')
ax.set_title('Power curve - Mean', fontweight='bold')

ax.grid(True, linestyle='--', alpha=0.4)

def validate(y, rounds=500, test_size=0.2):
    n = len(y)
    n_test = int(n * test_size)
    mse_scores = np.zeros(rounds)
    r2_scores  = np.zeros(rounds)
 
    for i in range(rounds):
        idx = np.random.permutation(n)
        idx_test, idx_train = idx[:n_test], idx[n_test:]
 
        y_train, y_test = y[idx_train], y[idx_test]
 
        y_pred = np.full(n_test, np.mean(y_train))
        
        mse_scores[i] = np.mean((y_test - y_pred) ** 2)
        r2_scores[i]  = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
 
    for name, scores in [('MSE', mse_scores), ('R2', r2_scores)]:
        print(f'{name} mean={scores.mean():.4f} std={scores.std():.4f}'
              f' min={scores.min():.4f} max={scores.max():.4f}')

validate(y)

# plt.tight_layout()
# plt.savefig('out/mean.png')
# plt.show()