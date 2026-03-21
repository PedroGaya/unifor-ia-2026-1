import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('data/EMGsDataset.csv', delimiter=',')
X = data[:2].T
y = data[2].astype(int)

C = y.max()
Y = np.zeros((len(y), C))
Y[np.arange(len(y)), y - 1] = 1

W = np.linalg.pinv(X) @ Y
 
y_hat = (X @ W).argmax(axis=1) + 1
accuracy = (y_hat == y).mean()
print(f'Accuracy: {accuracy:.2%}')

class_labels = {
    1: 'Neutro',
    2: 'Sorriso',
    3: 'Sobrancelhas levantadas',
    4: 'Surpreso',
    5: 'Rabugento',
}
colors = {1: 'green', 2: 'magenta', 3: 'red', 4: 'blue', 5: 'orange'}

fig, ax = plt.subplots(figsize=(10, 6))

# Decision regions via meshgrid
margin = 0.05
x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
pad1 = (x1_max - x1_min) * margin
pad2 = (x2_max - x2_min) * margin
xx1, xx2 = np.meshgrid(np.linspace(x1_min - pad1, x1_max + pad1, 1000),
                        np.linspace(x2_min - pad2, x2_max + pad2, 1000))
grid = np.c_[xx1.ravel(), xx2.ravel()]
zz = (grid @ W).argmax(axis=1) + 1
zz = zz.reshape(xx1.shape)
 
region_colors = [colors[c] for c in sorted(colors)]
ax.contourf(xx1, xx2, zz, levels=np.arange(0.5, C + 1.5),
            colors=region_colors, alpha=0.15)
ax.contour(xx1, xx2, zz, levels=np.arange(0.5, C + 1.5),
           colors='k', linewidths=0.8, linestyles='--', alpha=0.4)

rng = np.random.default_rng(42)
idx = rng.choice(len(y), size=min(2000, len(y)), replace=False)
X_plot, y_plot = X[idx], y[idx]

for cls in np.unique(y_plot):
    mask = y_plot == cls
    ax.scatter(X_plot[mask, 0], X_plot[mask, 1], c=colors[cls], label=class_labels[cls],
               s=60, alpha=0.6, edgecolors='white', linewidths=0.5)

ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_title(f'EMG', fontweight='bold')
ax.legend(title='Class')
ax.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig('out/emg/scatter_ols.png', dpi=150)
plt.show()