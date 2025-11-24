import numpy as np # Pour les opérations sur les tableaux ( pas de np.fft etc..)
from PIL import Image # Pour le chargement et la manipulation des images
import matplotlib.pyplot as plt # Librairie Tres utile pour afficher des graphiques
import cmath # Pour les nombres complexes
import math # Pour les fonctions mathématiques de base
import time # Pour mesurer le temps d'exécution

# FCTS DE TRANSFORMÉES DE FOURIER ( DISCRETE )

def dft_1d(x):
    x = np.asarray(x, dtype=complex)
    N = len(x)
    X = np.zeros(N, dtype=complex)

    for k in range(N):
        s = 0j
        for n in range(N):
            angle = -2 * math.pi * k * n / N
            s += x[n] * cmath.exp(1j * angle)
        X[k] = s

    return X


def idft_1d(X):
    X = np.asarray(X, dtype=complex)
    N = len(X)
    x = np.zeros(N, dtype=complex)

    for n in range(N):
        s = 0j
        for k in range(N):
            angle = 2 * math.pi * k * n / N
            s += X[k] * cmath.exp(1j * angle)
        x[n] = s / N

    return x


def dft_2d(img):
    img = np.asarray(img, dtype=complex)
    M, N = img.shape
    F = np.zeros((M, N), dtype=complex)

    for u in range(M):
        for v in range(N):
            s = 0j
            for x in range(M):
                for y in range(N):
                    angle = -2 * math.pi * ((u * x / M) + (v * y / N))
                    s += img[x, y] * cmath.exp(1j * angle)
            F[u, v] = s

    return F


def idft_2d(F):
    F = np.asarray(F, dtype=complex)
    M, N = F.shape
    img = np.zeros((M, N), dtype=complex)

    for x in range(M):
        for y in range(N):
            s = 0j
            for u in range(M):
                for v in range(N):
                    angle = 2 * math.pi * ((u * x / M) + (v * y / N))
                    s += F[u, v] * cmath.exp(1j * angle)
            img[x, y] = s / (M * N)

    return img.real


# FCTS DE TRANSFORMÉES DE FOURIER (recursive Cooley–Tukey)

def fft_1d(x):
    x = np.asarray(x, dtype=complex)
    N = len(x)

    if N == 1:
        return x

    even = fft_1d(x[::2])
    odd = fft_1d(x[1::2])

    X = np.zeros(N, dtype=complex)
    for k in range(N // 2):
        t = cmath.exp(-2j * math.pi * k / N) * odd[k]
        X[k] = even[k] + t
        X[k + N // 2] = even[k] - t

    return X


def ifft_1d(X):
    X = np.asarray(X, dtype=complex)
    N = len(X)

    conj = np.conjugate(X)
    y = fft_1d(conj)
    return np.conjugate(y) / N


def fft_2d(img):
    img = np.asarray(img, dtype=complex)

    # Rows
    temp = np.zeros_like(img, dtype=complex)
    for i in range(img.shape[0]):
        temp[i] = fft_1d(img[i])

    # Columns
    final = np.zeros_like(temp, dtype=complex)
    for j in range(img.shape[1]):
        final[:, j] = fft_1d(temp[:, j])

    return final


def ifft_2d(F):
    F = np.asarray(F, dtype=complex)

    # Rows
    temp = np.zeros_like(F, dtype=complex)
    for i in range(F.shape[0]):
        temp[i] = ifft_1d(F[i])

    # Columns
    final = np.zeros_like(temp, dtype=complex)
    for j in range(F.shape[1]):
        final[:, j] = ifft_1d(temp[:, j])

    return final.real



if __name__ == "__main__":

    # PART 1 — 1D SIGNAL

    print("\n===== 1D SIGNAL TEST =====")

    fs = 2048
    t = np.linspace(0, 1, fs)

    signal = np.sin(2 * np.pi * 10 * t) + 0.3 * np.sin(2 * np.pi * 55 * t)

    # ------- DFT 1D -------
    start = time.time()
    X_dft = dft_1d(signal)
    print(f"DFT 1D : {time.time() - start:.4f} s")

    # Remove 55 Hz
    X_dft[55] = 0
    X_dft[-55] = 0  # mirror frequency

    start = time.time()
    signal_rec_dft = idft_1d(X_dft)
    print(f"IDFT 1D : {time.time() - start:.4f} s")

    # ------- FFT 1D -------
    start = time.time()
    X_fft = fft_1d(signal)
    print(f"FFT 1D : {time.time() - start:.4f} s")

    X_fft[55] = 0
    X_fft[-55] = 0

    start = time.time()
    signal_rec_fft = ifft_1d(X_fft)
    print(f"IFFT 1D : {time.time() - start:.4f} s")

    # Plot 1D result
    plt.figure(figsize=(12, 5))
    plt.plot(t, signal, label="Signal sinusoïdale à 10 Hz avec une oscillation à 55 Hz", linewidth=1)
    plt.plot(t, signal_rec_fft.real, label="Bye Bye 55Hz", linewidth=1)
    plt.legend()
    plt.title("Signal 1D — Enlèvement d'une fréquence avec la FFT")
    plt.show()


  # ================================================
# PART 2 — 2D IMAGE (MEDIUM FREQUENCY REMOVAL)
# ================================================
print("\n===== 2D IMAGE TEST ( enlevement frequences medium ) =====")

img = Image.open("black_hole.jpg").convert("L")

img_small = img.resize((64, 64))
img_np = np.array(img_small, dtype=float)

# ------- DFT 2D -------
start = time.time()
F_dft = dft_2d(img_np)
print(f"DFT 2D : {time.time() - start:.4f} s")

H, W = F_dft.shape
cy, cx = H // 2, W // 2

low_cut = 5       # keep everything inside radius 6
high_cut = 35    # keep everything outside radius 14

for u in range(H):
    for v in range(W):
        dist = math.sqrt((u - cy)**2 + (v - cx)**2)
        if low_cut <= dist <= high_cut:   # remove only medium band
            F_dft[u, v] = 0

start = time.time()
img_rec_dft = idft_2d(F_dft)
print(f"IDFT 2D : {time.time() - start:.4f} s")

# ------- FFT 2D -------
start = time.time()
F_fft = fft_2d(img_np)
print(f"FFT 2D : {time.time() - start:.4f} s")

for u in range(H):
    for v in range(W):
        dist = math.sqrt((u - cy)**2 + (v - cx)**2)
        if low_cut <= dist <= high_cut:   # same mask
            F_fft[u, v] = 0

start = time.time()
img_rec_fft = ifft_2d(F_fft)
print(f"IFFT 2D : {time.time() - start:.4f} s")

# Show images
plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
plt.imshow(img_np, cmap="gray")
plt.title("Image (64×64)")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(np.log(1 + np.abs(F_fft)), cmap="gray")
plt.title("Spectre")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(img_rec_fft, cmap="gray")
plt.title("Médianes fréquences supprimées")
plt.axis("off")

plt.show()
