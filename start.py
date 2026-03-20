"""
Kernel Methods - Challenge Kaggle

Nathan Bouvier & Thomas Winninger

Ce code requière l'installation de torch, pandas et numpy.
"""

from dataclasses import dataclass
from typing import Annotated

import numpy as np
import pandas as pd
import torch as t
from numpy.typing import NDArray
from torch import Tensor

DEVICE = t.device("cuda" if t.cuda.is_available() else "cpu")
DATA_FOLDER = "data/"  # dossier contenant les images : DATA_FOLDER/Xte.csv, DATA_FOLDER/Xtr.csv, DATA_FOLDER/Ytr.csv


@dataclass
class Config:
    C: float = 10.0  # change pas grand chose
    n_iter: int = 2000  # fixé
    n_pca: int | None = (
        800  # important -> moins de dim moins d'info mais plus facile à apprendre
    )


config = Config()


def compute_kernel(
    X: Annotated[Tensor, "n d"], Y: Annotated[Tensor, "m d"]
) -> Annotated[Tensor, "n m"]:
    return (0.01 * (X @ Y.T) + 1.0).clamp_(min=0.0) ** 2


def project_alpha(
    v: Annotated[Tensor, "n k"], Delta: Annotated[Tensor, "n k"], C: float
) -> Annotated[Tensor, "n k"]:
    """
    Projection de alpha sur l'ensemble des contraintes avec la méthode des bisections.
    """
    # Interval de recherche
    mu_min = v.min(dim=1, keepdim=True)[0] - C
    mu_max = v.max(dim=1, keepdim=True)[0]

    for _ in range(30):
        mu = (mu_min + mu_max) / 2.0
        u = t.min(v - mu, C * Delta)
        sum_u = u.sum(dim=1, keepdim=True)

        mu_min = t.where(sum_u > 0, mu, mu_min)
        mu_max = t.where(sum_u > 0, mu_max, mu)

    return t.min(v - mu_max, C * Delta)


def train(
    K_train: Annotated[Tensor, "n n"],
    Y_train: Annotated[Tensor, "n"],
    n_iter: int,
    C: float,
    k_classes: int = 10,
) -> Annotated[Tensor, "n k"]:
    n = K_train.shape[0]

    Delta = t.zeros((n, k_classes), device=K_train.device)
    Delta[t.arange(n), Y_train] = 1.0

    # -- Calcul de la constante de Lipschitz pour trouver le pas d'apprentissage optimal --
    v = t.randn(n, 1, device=K_train.device)
    v /= v.norm()
    for _ in range(20):
        v = K_train @ v
        lam = v.norm()
        v /= lam

    lr = 0.9 / lam.item()
    alpha = t.zeros((n, k_classes), device=K_train.device)

    # -- Montée de gradient avec projection à chaque étape --
    for _ in range(n_iter):
        grad = Delta - K_train @ alpha
        alpha_unprojected = alpha + lr * grad
        alpha = project_alpha(alpha_unprojected, Delta, C)

    return alpha


def predict(
    K_test: Annotated[Tensor, "m n"],
    alpha: Annotated[Tensor, "n k"],
) -> Annotated[NDArray, "m"]:
    scores = K_test @ alpha
    return scores.argmax(dim=1).cpu().numpy()


## -- Preprocessing (HOG + LBP + PCA) --
def hog_features(
    channel: Annotated[Tensor, "h w"],
    pixels_per_cell: int = 8,
    n_bins: int = 9,
    cell_per_block: int = 2,
) -> Annotated[NDArray, "n_hog_features"]:
    """
    Extraction des caractéristiques HOG pour une image et une couleur : taille (height, width) dans [0, 1].

    Détails:
    - Le gradient au bord est mis à 0
    - L'orientation est calculée via arctan au lieu de abs(). Elle appartient à [0, 180) car on considère généralement les directions opposées comme égales.
    - La norme utilisée pour chaque bloc est L2-Hys : L2, tronqué à 0.2, L2.
    """
    image = np.asarray(channel, dtype=np.float64)
    height, width = image.shape

    # -- Calcul des gradients --
    grad_x = np.zeros_like(image)
    grad_y = np.zeros_like(image)
    grad_x[:, 1:-1] = image[:, 2:] - image[:, :-2]
    grad_y[1:-1, :] = image[2:, :] - image[:-2, :]

    # -- Amplitude et angle de chaque bloc --
    magnitudes = np.hypot(grad_x, grad_y)
    angles = np.rad2deg(np.arctan2(grad_y, grad_x)) % 180.0

    # -- Division en cellules --
    n_cells_y, n_cells_x = height // pixels_per_cell, width // pixels_per_cell
    cell_magnitudes = (
        magnitudes[: n_cells_y * pixels_per_cell, : n_cells_x * pixels_per_cell]
        .reshape(n_cells_y, pixels_per_cell, n_cells_x, pixels_per_cell)
        .transpose(0, 2, 1, 3)
        .reshape(n_cells_y, n_cells_x, -1)
    )
    cell_angles = (
        angles[: n_cells_y * pixels_per_cell, : n_cells_x * pixels_per_cell]
        .reshape(n_cells_y, pixels_per_cell, n_cells_x, pixels_per_cell)
        .transpose(0, 2, 1, 3)
        .reshape(n_cells_y, n_cells_x, -1)
    )

    # -- Création de l'histogramme : 180 deg répartis sur n_bins --
    bin_width = 180.0 / n_bins
    cont_idx = cell_angles / bin_width - 0.5
    bin_1 = np.floor(cont_idx).astype(np.int32)
    weight_2 = (cont_idx - bin_1).astype(np.float32)
    weight_1 = 1.0 - weight_2
    bin_1 = bin_1 % n_bins
    bin_2 = (bin_1 + 1) % n_bins

    cell_hist = np.zeros((n_cells_y, n_cells_x, n_bins), dtype=np.float32)
    for bin_ in range(n_bins):
        cell_hist[:, :, bin_] = np.where(
            bin_1 == bin_, weight_1 * cell_magnitudes, 0.0
        ).sum(axis=-1) + np.where(bin_2 == bin_, weight_2 * cell_magnitudes, 0.0).sum(
            axis=-1
        )

    # -- Normalisation des blocs --
    eps = 1e-5
    n_bins_y = n_cells_y - cell_per_block + 1
    n_bins_x = n_cells_x - cell_per_block + 1
    out = np.empty(
        n_bins_y * n_bins_x * cell_per_block * cell_per_block * n_bins, dtype=np.float32
    )
    k = 0
    for bin_y in range(n_bins_y):
        for bin_x in range(n_bins_x):
            block = (
                cell_hist[
                    bin_y : bin_y + cell_per_block, bin_x : bin_x + cell_per_block
                ]
                .ravel()
                .astype(np.float64)
            )

            block = block / np.sqrt((block**2).sum() + eps**2)
            block = np.clip(block, 0.0, 0.2)
            block = block / np.sqrt((block**2).sum() + eps**2)
            size = block.size
            out[k : k + size] = block
            k += size
    return out


def lbp_features(
    gray: Annotated[NDArray, "h w"], radius: int, n_points: int
) -> Annotated[NDArray, "n_lpb_features"]:
    """
    Calcul des caractéristiques LPB de chaque patch.

    Détails:
    - Utilisation d'un warp aux bordures, n'a pas l'air de poser problème.
    """
    angles = 2 * np.pi * np.arange(n_points) / n_points
    dy = -np.round(radius * np.sin(angles)).astype(np.int32)
    dx = np.round(radius * np.cos(angles)).astype(np.int32)

    n = 1 << n_points
    codes = np.arange(n, dtype=np.int32)
    bits = (codes[:, None] >> np.arange(n_points)[None, :]) & 1  # (n, p)
    transitions = (bits != np.roll(bits, -1, axis=1)).sum(axis=1)
    label = bits.sum(axis=1)
    lookup_table = np.where(transitions <= 2, label, n_points + 1).astype(np.int32)

    gray32 = gray.astype(np.int32)
    pattern = np.zeros_like(gray32)
    for p in range(n_points):
        neighbor = np.roll(np.roll(gray32, int(dy[p]), axis=0), int(dx[p]), axis=1)
        pattern |= (neighbor >= gray32).astype(np.int32) << p

    n_bins = n_points + 2
    mapped = lookup_table[pattern]
    histogram = np.bincount(mapped.ravel(), minlength=n_bins).astype(np.float32)
    return histogram / (np.linalg.norm(histogram) + 1e-7)


def extract_features(X: Annotated[NDArray, "n d_raw"]) -> Annotated[NDArray, "n d"]:
    """
    Extracttion des caractéristiques HOG, LBP, et histogrammes des couleurs : 6500 features.
    """
    N = len(X)
    images = X.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    features = []
    for i in range(N):
        if i % 100 == 0:
            print(f"Image {i}/{N}")
        image = images[i].astype(np.float32)
        image_norm = (image - image.min()) / (image.max() - image.min() + 1e-8)

        image_features = []

        # -- HOG --
        for pixel_per_cell in [8, 4]:
            for channel in range(3):
                image_features.append(
                    hog_features(
                        image_norm[:, :, channel], pixels_per_cell=pixel_per_cell
                    )
                )

        # -- Couleurs (32 bins) --
        for channel in range(3):
            h, _ = np.histogram(image_norm[:, :, channel], bins=32, range=(0.0, 1.0))
            image_features.append(h.astype(np.float32))

        # -- LBP --
        gray = (
            0.2126 * image_norm[:, :, 0]
            + 0.7152 * image_norm[:, :, 1]
            + 0.0722 * image_norm[:, :, 2]
        )  # valeurs récupérées sur skimage
        gray_u8 = (gray * 255).astype(np.uint8)
        H, W = gray_u8.shape

        ## Choix des régions : toute l'image, puis les 4 coins
        for radius, n_points in [(1, 8), (2, 16)]:
            regions = [
                gray_u8,
                gray_u8[: H // 2, : W // 2],
                gray_u8[: H // 2, W // 2 :],
                gray_u8[H // 2 :, : W // 2],
                gray_u8[H // 2 :, W // 2 :],
            ]
            for region in regions:
                image_features.append(lbp_features(region, radius, n_points))

        features.append(np.concatenate(image_features))

    return np.array(features, dtype=np.float32)


def standardize(
    X_train: Annotated[NDArray, "n d"], X_test: Annotated[NDArray, "m d"]
) -> tuple[Annotated[NDArray, "n d"], Annotated[NDArray, "m d"]]:
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    return ((X_train - mean) / std).astype(np.float32), ((X_test - mean) / std).astype(
        np.float32
    )


def apply_pca(
    X_train: Annotated[NDArray, "n d"],
    X_test: Annotated[NDArray, "m d"],
    n_components: int,
) -> tuple[Annotated[NDArray, "n d_pca"], Annotated[NDArray, "m d_pca"]]:
    mean = X_train.mean(axis=0)
    Xc = X_train - mean
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    V = Vt[:n_components].T
    var = (s[:n_components] ** 2).sum() / (s**2).sum()
    print(
        f"PCA: réduction de {X_train.shape[1]} à {n_components} dimensions, variance expliquée : {var:.1%}."
    )
    return ((X_train - mean) @ V).astype(np.float32), ((X_test - mean) @ V).astype(
        np.float32
    )


## -- Run --


config = Config()

print("-- Démarrage --")
Xtr = np.array(
    pd.read_csv(f"{DATA_FOLDER}/Xtr.csv", header=None, sep=",", usecols=range(3072))
)
Xte = np.array(
    pd.read_csv(f"{DATA_FOLDER}/Xte.csv", header=None, sep=",", usecols=range(3072))
)
Ytr = np.array(pd.read_csv(f"{DATA_FOLDER}/Ytr.csv", sep=",", usecols=[1])).squeeze()

print(f"Train: {Xtr.shape}, test: {Xte.shape}, classes: {np.unique(Ytr)}")

print("-- Extraction des caractéristiques --")
F_tr = extract_features(Xtr)
F_te = extract_features(Xte)

print("-- Normalisation -- ")
F_tr, F_te = standardize(F_tr, F_te)
F_tr, F_te = apply_pca(F_tr, F_te, n_components=config.n_pca)

X_tr = t.tensor(F_tr, dtype=t.float32, device=DEVICE)
X_te = t.tensor(F_te, dtype=t.float32, device=DEVICE)
K_tr = compute_kernel(X_tr, X_tr)
K_te = compute_kernel(X_te, X_tr)

print("-- Apprentissage --")
alphas = train(K_tr, Ytr, n_iter=config.n_iter, C=config.C)

print("-- Prédiction --")
y_pred = predict(K_te, alphas)
df = pd.DataFrame({"Id": np.arange(1, len(y_pred) + 1), "Prediction": y_pred})
df.to_csv("Yte.csv", index=False)

print("-- Prédictions sauvegardées dans Yte.csv --")
