import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('data/EMGsDataset.csv', delimiter=',')

X1 = data[0]
X2 = data[1]
y  = data[2].astype(int)

class_labels = {
    1: 'Neutro',
    2: 'Sorriso',
    3: 'Sobrancelhas levantadas',
    4: 'Surpreso',
    5: 'Rabugento',
}
colors = {1: 'green', 2: 'magenta', 3: 'red', 4: 'blue', 5: 'orange'}

fig, ax = plt.subplots(figsize=(10, 6))

for cls in np.unique(y):
    mask = y == cls
    ax.scatter(X1[mask], X2[mask], c=colors[cls], label=class_labels[cls], 
               s=30, alpha=0.6, edgecolors='white', linewidths=0.5)

ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_title('EMG Facial Sensor Data', fontweight='bold')

ax.legend(title='Class')
ax.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig('out/scatter_emg.png')
plt.show()