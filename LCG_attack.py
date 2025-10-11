import numpy as np
import time
from math import gcd
import matplotlib.pyplot as plt

# ----------------------------
# LCG definition
# ----------------------------
class LCG:
    def __init__(self, seed, a, b, M):
        self.a = a
        self.b = b
        self.M = M
        self.state = seed % M

    def next_int(self):
        self.state = (self.a * self.state + self.b) % self.M
        return self.state

    def generate(self, N):
        return [self.next_int() for _ in range(N)]

# ----------------------------
# Modular inverse
# ----------------------------
def modinv(x, M):
    g, inv, _ = extended_gcd(x, M)
    if g != 1:
        return None
    return inv % M

def extended_gcd(a, b):
    if b == 0:
        return a, 1, 0
    else:
        g, x1, y1 = extended_gcd(b, a % b)
        x = y1
        y = x1 - (a // b) * y1
        return g, x, y

# ----------------------------
# Recover (a,b) from triple
# ----------------------------
def recover_ab_from_triple(xn, xn1, xn2, M):
    d1 = (xn1 - xn) % M
    d2 = (xn2 - xn1) % M
    g = gcd(d1, M)
    candidates = []
    if g == 1:
        a = (d2 * modinv(d1, M)) % M
        b = (xn1 - a * xn) % M
        candidates.append((a, b))
    return candidates, g

# ----------------------------
# Parameters
# ----------------------------
seed = 16384
a_true = 1679
b_true = 1409
M = 32768
N = 200  # longer sequence for more triples

# Generate LCG sequence
lcg = LCG(seed, a_true, b_true, M)
X = lcg.generate(N)

# ----------------------------
# Analyze triples
# ----------------------------
success_count = 0
failure_count = 0
invertible_count = 0
non_invertible_count = 0
recovery_times = []

for i in range(N-2):
    triple = X[i:i+3]
    start_time = time.time()
    candidates, g = recover_ab_from_triple(*triple, M)
    elapsed = time.time() - start_time
    if g == 1:
        invertible_count += 1
        # Check if true (a,b) is found
        valid = any([c[0] == a_true and c[1] == b_true for c in candidates])
        if valid:
            success_count += 1
            recovery_times.append(elapsed)
    else:
        non_invertible_count += 1
        failure_count += 1

# ----------------------------
# Summary statistics
# ----------------------------
total_triples = N - 2
frac_invertible = invertible_count / total_triples

# Print table-like summary
print(f"{'Metric':<30} {'Value':<15}")
print(f"{'-'*45}")
print(f"{'Total triples':<30} {total_triples}")
print(f"{'Invertible triples (gcd=1)':<30} {invertible_count}")
print(f"{'Non-invertible triples (gcd>1)':<30} {non_invertible_count}")
print(f"{'Successful recovery':<30} {success_count}")
print(f"{'Failed recovery':<30} {failure_count}")
print(f"{'Fraction invertible':<30} {frac_invertible:.3f}")
if recovery_times:
    print(f"{'Mean recovery time [s]':<30} {np.mean(recovery_times):.6f}")
    print(f"{'Max recovery time [s]':<30} {np.max(recovery_times):.6f}")

# ----------------------------
# Histogram of recovery times
# ----------------------------
plt.figure(figsize=(8,5))
plt.hist(recovery_times, bins=20, color='black', alpha=0.7, edgecolor='white')
plt.xlabel("Recovery time [s]")
plt.ylabel("Frequency")
plt.title("Distribution of Recovery Times for Invertible Triples")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
