import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('data/aerogerador.dat')
X = data[:,0]
y = data[:,1]
deg = 5

X_poly = np.column_stack([X ** d for d in range(deg + 1)])
beta_hat = np.linalg.inv(X_poly.T @ X_poly) @ X_poly.T @ y

x_line = np.linspace(X.min(), X.max(), 200)
x_poly_line = np.column_stack([x_line ** d for d in range(deg + 1)])
y_line = x_poly_line @ beta_hat

fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(X, y)
ax.plot(x_line, y_line, color='red')

ax.set_xlabel('Wind speed (m/s)')
ax.set_ylabel('Power (kW)')
ax.set_title('Power curve - OLS', fontweight='bold')

ax.grid(True, linestyle='--', alpha=0.4)

def validate(X, y, degree=3, rounds=500, test_size=0.2):
    n = len(y)
    n_test = int(n * test_size)
    mse_scores = np.zeros(rounds)
    r2_scores  = np.zeros(rounds)

    for i in range(rounds):
        idx = np.random.permutation(n)
        idx_test, idx_train = idx[:n_test], idx[n_test:]

        X_train, y_train = X[idx_train], y[idx_train]
        X_test,  y_test  = X[idx_test],  y[idx_test]

        # Fit on raw X, build poly features inside
        X_train_poly = np.column_stack([X_train ** d for d in range(degree + 1)])
        X_test_poly  = np.column_stack([X_test ** d for d in range(degree + 1)])

        b = np.linalg.inv(X_train_poly.T @ X_train_poly) @ X_train_poly.T @ y_train
        y_pred = X_test_poly @ b

        mse_scores[i] = np.mean((y_test - y_pred) ** 2)
        r2_scores[i]  = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)

    print(f'Degree: {degree}')
    for name, scores in [('MSE', mse_scores), ('R2', r2_scores)]:
        print(f'  {name} mean={scores.mean():.4f} std={scores.std():.4f}'
              f' min={scores.min():.4f} max={scores.max():.4f}')

validate(X, y, degree=5)

plt.tight_layout()
plt.savefig(f'out/aero/polyfit_{deg}deg.png')
plt.show()