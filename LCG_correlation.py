import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- LCG class ---
class LinearCongruentialGenerator:
    def __init__(self, seed: int, a: int, b: int, m: int):
        self.m = int(m)
        self.a = int(a)
        self.b = int(b)
        self.seed = int(seed) % self.m
        self.state = int(self.seed)
    
    def next_int(self) -> int:
        self.state = (self.a * self.state + self.b) % self.m
        return self.state
    
    def generate_ints(self, n: int):
        return [self.next_int() for _ in range(n)]
    
    def reset(self):
        self.state = int(self.seed)

# -------------------------
# Parameters
# -------------------------
m = 100_000         # large modulus
seed = 1
seq_len = 32
num_samples = 500   # sample number of a and b values

# Randomly sample a and b values
np.random.seed(42)
a_vals = np.random.randint(1, m, size=num_samples)
b_vals = np.random.randint(0, m, size=num_samples)

corr_matrix = np.zeros((num_samples, num_samples))

# -------------------------
# Compute correlation
# -------------------------
for i, a in enumerate(a_vals):
    for j, b in enumerate(b_vals):
        lcg = LinearCongruentialGenerator(seed, a, b, m)
        x = np.array(lcg.generate_ints(seq_len))
        if np.std(x) == 0:
            corr_matrix[i, j] = 1.0
        else:
            corr_matrix[i, j] = np.corrcoef(x[:-1], x[1:])[0,1]

# -------------------------
# Plot 1: 3D Surface
# -------------------------
A, B = np.meshgrid(b_vals, a_vals)
fig1 = plt.figure(figsize=(12,7))
ax1 = fig1.add_subplot(111, projection='3d')
surf = ax1.plot_surface(A, B, corr_matrix, cmap='viridis', alpha=0.8)

# Zero-correlation plane
z_plane = np.zeros_like(A)
ax1.plot_surface(A, B, z_plane, color='gray', alpha=0.2)

# Highlight points near zero correlation
tol = 0.02
zero_idx = np.abs(corr_matrix) < tol
ax1.scatter(A[zero_idx], B[zero_idx], corr_matrix[zero_idx],
            color='red', s=20, label='Correlation ≈ 0')

ax1.set_xlabel("b (increment)")
ax1.set_ylabel("a (multiplier)")
ax1.set_zlabel("Lag-1 Correlation")
ax1.set_title(f"LCG Parameter Sweep: Zero-Correlation Plane (m={m})")
ax1.view_init(elev=30, azim=45)
fig1.colorbar(surf, shrink=0.5, aspect=5)
ax1.legend()

# -------------------------
# Plot 2: 2D Zero-Correlation Slice
# -------------------------
fig2, ax2 = plt.subplots(figsize=(8,6))
zero_mask = np.abs(corr_matrix) < tol
ax2.scatter(A[zero_mask], B[zero_mask], color='red', s=20)

ax2.set_xlabel("b (increment)")
ax2.set_ylabel("a (multiplier)")
ax2.set_title(f"LCG Zero-Correlation Points (m={m})")
ax2.grid(True)
plt.show()
