import numpy as np
import matplotlib.pyplot as plt

# Dados são da forma: m/s kWh
data = np.loadtxt('data/aerogerador.dat')

X = data[:, 0]
y   = data[:, 1]

fig, ax = plt.subplots(figsize=(10, 6))

scatter = ax.scatter(X, y)

ax.set_xlabel('Wind speed (m/s)')
ax.set_ylabel('Power (kW)')
ax.set_title('Power curve for wind turbine', fontweight='bold')

ax.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig('out/scatter.png')
plt.show()