#!/usr/bin/env python3
"""
lcg_stride_full_recovery_fixed_corr.py

Experiment demonstrating:
 - stride = 0 : degenerate case, repeated values, no info
 - stride = 1 : consecutive samples -> perfect recovery
 - stride > 1 : non-consecutive samples -> attempt to recover M, a, b
Now also computes and plots cross-correlation between original and recovered sequences.
"""

import time
from math import gcd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# LCG definition
# -------------------------
class LCG:
    def __init__(self, seed, a, b, M):
        self.a = int(a)
        self.b = int(b)
        self.M = int(M)
        self.state = int(seed) % self.M

    def next_int(self):
        self.state = (self.a * self.state + self.b) % self.M
        return self.state

    def generate(self, N):
        return [self.next_int() for _ in range(int(N))]

# -------------------------
# Extended GCD and modular inverse
# -------------------------
def extended_gcd(a, b):
    if b == 0:
        return (abs(a), 1 if a >= 0 else -1, 0)
    g, x1, y1 = extended_gcd(b, a % b)
    x = y1
    y = x1 - (a // b) * y1
    return (g, x, y)

def modinv(x, M):
    g, inv, _ = extended_gcd(x % M, M)
    if g != 1:
        return None
    return inv % M

# -------------------------
# Estimate modulus from sequence
# -------------------------
def estimate_modulus_from_sequence(X):
    """Estimate modulus M from sequence X using the gcd trick on differences."""
    if len(X) < 4:
        return None
    diffs = [int(X[i+1] - X[i]) for i in range(len(X)-1)]
    Ys = []
    for i in range(len(diffs)-2):
        val = diffs[i+2]*diffs[i] - (diffs[i+1]**2)
        if val != 0:
            Ys.append(abs(val))
    if not Ys:
        return None
    M_est = Ys[0]
    for v in Ys[1:]:
        M_est = gcd(M_est, v)
    return abs(M_est) if M_est != 0 else None

# -------------------------
# Recover a,b from triple and known M
# -------------------------
def recover_ab_from_triple(xn, xnp1, xnp2, M):
    d1 = (xnp1 - xn) % M
    d2 = (xnp2 - xnp1) % M
    g = gcd(d1, M)
    if d1 == 0 or (d2 % g) != 0:
        return []
    M1 = M // g
    d1r = d1 // g
    d2r = d2 // g
    inv = modinv(d1r, M1)
    if inv is None:
        return []
    a0 = (d2r * inv) % M1
    candidates = []
    for t in range(g):
        a_cand = (a0 + t * M1) % M
        b_cand = (xnp1 - a_cand * xn) % M
        candidates.append((a_cand, b_cand))
    return candidates

# -------------------------
# Recover a,b specifically for stride=1
# -------------------------
def recover_ab_stride1(x0, x1, x2, M):
    d1 = (x1 - x0) % M
    if d1 == 0:
        return []
    inv = modinv(d1, M)
    if inv is None:
        return []
    a = ((x2 - x1) * inv) % M
    b = (x1 - a * x0) % M
    return [(a, b)]

# -------------------------
# Cross-correlation computation
# -------------------------
def normalized_cross_correlation(x, y):
    """Return normalized cross-correlation between two sequences."""
    x = np.array(x) - np.mean(x)
    y = np.array(y) - np.mean(y)
    corr = np.correlate(x, y, mode='full')
    corr /= np.sqrt(np.sum(x**2) * np.sum(y**2))
    lags = np.arange(-len(x)+1, len(x))
    return lags, corr

# -------------------------
# Main experiment
# -------------------------
def run_experiment(seed=16384, a=1679, b=1409, M=32768, N=20000, max_stride=10):
    print(f"Generating LCG (seed={seed}, a={a}, b={b}, M={M}, N={N})")
    lcg = LCG(seed, a, b, M)
    full_seq = lcg.generate(N)

    stride_stats = []

    for s in range(0, max_stride+1):
        print(f"\n--- Stride s={s} ---")
        M_est = None
        a_found = None
        b_found = None
        found = False
        match_frac = 0.0
        recovery_time = 0.0
        Xr = None

        if s == 0:
            print("Stride=0: attacker sees repeated values — no info.")
            stride_stats.append((s, {'found': found,'match_frac': match_frac,'recovery_time': recovery_time,'M_est': M_est,'a_found': a_found,'b_found': b_found}))
            continue

        if 2*s >= N:
            print("Not enough data for this stride, skipping.")
            stride_stats.append((s, {'found': False,'match_frac': 0.0,'recovery_time': 0.0,'M_est': None,'a_found': None,'b_found': None}))
            continue

        sampled = [full_seq[i] for i in [0, s, 2*s]]
        start = time.time()

        M_est = estimate_modulus_from_sequence(full_seq[:min(500, N)]) or M

        if s == 1:
            cands = recover_ab_stride1(sampled[0], sampled[1], sampled[2], M_est)
        else:
            cands = recover_ab_from_triple(sampled[0], sampled[1], sampled[2], M_est)

        if cands:
            a_s, b_s = cands[0]
            if s == 1:
                a_try, b_try = a_s, b_s
                ok = True
                x = full_seq[0]
                for i in range(1, min(200, len(full_seq))):
                    x = (a_try * x + b_try) % M_est
                    if x != full_seq[i]:
                        ok = False
                        break
                if ok:
                    found = True
                    a_found = a_try
                    b_found = b_try
            else:
                a_candidates = [a_try for a_try in range(M_est) if pow(a_try, s, M_est) == a_s]
                for a_try in a_candidates:
                    b_try = (full_seq[1] - a_try * full_seq[0]) % M_est
                    x = full_seq[0]
                    ok = True
                    for i in range(1, min(200, len(full_seq))):
                        x = (a_try * x + b_try) % M_est
                        if x != full_seq[i]:
                            ok = False
                            break
                    if ok:
                        found = True
                        a_found = a_try
                        b_found = b_try
                        break

            if found:
                rec = LCG(full_seq[0], a_found, b_found, M_est)
                Xr = rec.generate(N)
                X_true = np.array(full_seq)
                match_frac = float(np.mean(np.array(Xr) == X_true))

        end = time.time()
        recovery_time = end - start

        print(f"Original params: M={M}, a={a}, b={b}")
        print(f"Estimated params: M_est={M_est}, a_found={a_found}, b_found={b_found}")
        print("First 20 original x(n):", full_seq[:20])
        if found and Xr is not None:
            print("First 20 recovered x(n):", Xr[:20])
        else:
            print("Recovered sequence: <not available / recovery failed>")

        print(f"Stride {s}: found={found}, match_frac={match_frac:.4f}, recovery_time={recovery_time:.4f}s")

        stride_stats.append((s, {'found': found,'match_frac': match_frac,'recovery_time': recovery_time,'M_est': M_est,'a_found': a_found,'b_found': b_found}))

        # ---- Cross-correlation plot ----
        if found and Xr is not None:
            lags, corr = normalized_cross_correlation(full_seq, Xr)
            lag_max = lags[np.argmax(corr)]
            corr_max = np.max(corr)
            print(f"Cross-corr peak: {corr_max:.4f} at lag={lag_max}")
            plt.figure(figsize=(8,5))
            plt.plot(lags, corr)
            plt.title(f"Cross-correlation (Original vs Recovered) — Stride s={s}")
            plt.xlabel("Lag")
            plt.ylabel("Normalized correlation")
            plt.grid(True)
            plt.show()

    # ---- Accuracy and time plots ----
    strides = [s for s, _ in stride_stats]
    matches = [st['match_frac'] for _, st in stride_stats]
    times = [st['recovery_time'] for _, st in stride_stats]

    plt.figure(figsize=(8,5))
    plt.plot(strides, matches, marker='o')
    plt.xlabel("Stride s")
    plt.ylabel("Prediction Accuracy")
    plt.title("Accuracy vs Stride (including s=0)")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8,5))
    plt.plot(strides, times, marker='o', color='orange')
    plt.xlabel("Stride s")
    plt.ylabel("Recovery Time (s)")
    plt.title("Recovery Time vs Stride")
    plt.grid(True)
    plt.show()

    return stride_stats

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    stats = run_experiment(seed=16384, a=1679, b=1409, M=32768, N=20000, max_stride=10)
    print("\nSummary:")
    for s, st in stats:
        print(f"s={s}: found={st['found']}, match={st['match_frac']:.4f}, "
              f"M_est={st['M_est']}, a={st['a_found']}, b={st['b_found']}, time={st['recovery_time']:.4f}s")
