import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.metrics import mutual_info_score

# -------------------------------
# PLOT SETTINGS (IEEE STYLE)
# -------------------------------
FONT_FAMILY = "Arial"
AXIS_TITLE_SIZE = 20
TICK_LABEL_SIZE = 16
LINE_WIDTH = 1.2
MARKER_SIZE = 3

plt.rcParams.update({
    "font.family": FONT_FAMILY,
    "axes.titlesize": AXIS_TITLE_SIZE,
    "axes.labelsize": AXIS_TITLE_SIZE,
    "xtick.labelsize": TICK_LABEL_SIZE,
    "ytick.labelsize": TICK_LABEL_SIZE,
    "axes.linewidth": 1.2,
    "lines.linewidth": LINE_WIDTH,
    "lines.markersize": MARKER_SIZE,
    "grid.linewidth": 0.6,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
})

# -------------------------------
# LCG PARAMETERS
# -------------------------------
a = 1679
b = 1409
M = 32768
seed = 16384
N = 4096  # number of samples

# -------------------------------
# LCG GENERATOR
# -------------------------------
def lcg(a, b, M, seed, N):
    x = np.zeros(N, dtype=int)
    x[0] = seed % M
    for i in range(1, N):
        x[i] = (a * x[i-1] + b) % M
    return x

x = lcg(a, b, M, seed, N)
x_norm = x / M  # normalize between 0–1

# -------------------------------
# 1️⃣ TIME SERIES (FIRST 256)
# -------------------------------
plt.figure(figsize=(6,3))
plt.plot(x_norm[:256])
plt.xlabel("Sample index (n)")
plt.ylabel("Normalized value")
plt.title("LCG Time Series (first 256 samples)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# -------------------------------
# 2️⃣ HISTOGRAM (VALUE DISTRIBUTION)
# -------------------------------
plt.figure(figsize=(4.5,3))
plt.hist(x_norm, bins=50, density=True, alpha=0.7, edgecolor='black')
plt.xlabel("Value")
plt.ylabel("Probability Density")
plt.title("Histogram of LCG Values")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# -------------------------------
# 3️⃣ AUTOCORRELATION FUNCTION (ACF)
# -------------------------------
def autocorr(x, lag):
    n = len(x)
    x_mean = np.mean(x)
    c = np.correlate(x - x_mean, x - x_mean, mode='full')
    c = c[c.size//2:] / c[c.size//2]
    return c[:lag]

lags = 128
acf_vals = autocorr(x_norm, lags)
plt.figure(figsize=(6,3))
markerline, stemlines, baseline = plt.stem(range(lags), acf_vals)
plt.setp(stemlines, linewidth=1.2)
plt.setp(markerline, markersize=3)
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.title("Autocorrelation Function (up to lag 128)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()


# -------------------------------
# 4️⃣ POWER SPECTRUM (FFT)
# -------------------------------
X = np.fft.fft(x_norm - np.mean(x_norm))
freqs = np.fft.fftfreq(len(X), d=1.0)
pos = freqs >= 0
plt.figure(figsize=(6,3))
plt.plot(freqs[pos], np.abs(X[pos])/len(X))
plt.xlabel("Normalized Frequency")
plt.ylabel("Magnitude")
plt.title("Power Spectrum (FFT magnitude)")
plt.xlim(0, 0.5)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# -------------------------------
# 5️⃣ CHI-SQUARE TEST (UNIFORMITY)
# -------------------------------
hist, bins = np.histogram(x_norm, bins=20)
expected = np.ones_like(hist) * np.mean(hist)
chi2, p, dof, exp = chi2_contingency([hist, expected])
print(f"Chi-square test p-value for uniformity: {p:.4f}")

# -------------------------------
# 6️⃣ MUTUAL INFORMATION (x[n], x[n+1])
# -------------------------------
mi = mutual_info_score(None, None, contingency=np.histogram2d(
    x_norm[:-1], x_norm[1:], bins=50)[0])
print(f"Mutual Information between consecutive samples: {mi:.6f} bits")
