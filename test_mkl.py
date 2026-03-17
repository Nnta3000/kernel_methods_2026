import json
import numpy as np
import jax.numpy as jnp
from jaxtyping import Array, Float
import kernels 
from classifiers import KernelRidgeRegression
from utils.classification import OneVsAllStrategy, Trainer, Regularizer
from utils.kernelisation import KernelDataset, KernelDataLoader
from utils.tools import prediction_to_csv
from gradient_ascent import build_kernel

MIN_SCORE = 0.0


def load_good_configs(path: str = "kernel_results.json") -> list[dict]:
    with open(path) as f:
        results = json.load(f)
    good = [r for r in results if r["score"] >= MIN_SCORE]
    print(f"Loaded {len(good)}/{len(results)} configs with score >= {MIN_SCORE}")
    for r in good:
        print(f"  {r['algo']} + {r['kernel']} — score={r['score']:.4f} params={r['params']}")
    return good


class MKLTrainer:
    """
    Multiple Kernel Learning: combines kernel matrices as a weighted sum
    K_mkl = sum_i w_i * K_i, then runs KRR/KLR on the combined kernel.
    Weights are optimized by gradient ascent on val accuracy.
    """
    def __init__(self, kernel_matrices: list[jnp.ndarray], kernel_matrices_val: list[jnp.ndarray],
                 y_train: jnp.ndarray, y_val: jnp.ndarray):
        self.Ks_train = kernel_matrices
        self.Ks_val   = kernel_matrices_val
        self.y_train  = y_train
        self.y_val    = y_val

    def combined(self, weights: np.ndarray) -> tuple[Float[Array, ""], Float[Array, ""]]:
        w = jnp.array(weights / weights.sum())  # normalize
        K_train:Float[Array, ""] = sum(w[i] * self.Ks_train[i] for i in range(len(w)))
        K_val: Float[Array, ""]    = sum(w[i] * self.Ks_val[i]   for i in range(len(w)))
        return K_train, K_val

    def score(self, weights: np.ndarray, lam: float) -> float:
        K_train, K_val = self.combined(weights)
        reg = Regularizer(lam=lam)
        strategy = OneVsAllStrategy(n_classes=10, model_factory=lambda: KernelRidgeRegression(reg))
        strategy.fit(K_train, self.y_train)
        preds = strategy.predict(K_val)
        return float(jnp.mean(preds == self.y_val))

    def optimize_weights(self, lam: float, n_iter: int = 20, lr: float = 0.3, eps: float = 0.2) -> tuple[float, np.ndarray]:
        n = len(self.Ks_train)
        log_w = np.zeros(n)  # start uniform in log space
        best_score = self.score(np.exp(log_w), lam)
        print(f"[Init] score={best_score:.4f} weights={np.exp(log_w) / np.exp(log_w).sum()}")

        for i in range(n_iter):
            grads = np.zeros(n)
            for j in range(n):
                w_up, w_down = np.copy(log_w), np.copy(log_w)
                w_up[j]   += eps
                w_down[j] -= eps
                grads[j]   = (self.score(np.exp(w_up), lam) - self.score(np.exp(w_down), lam)) / (2 * eps)
            log_w += lr * grads
            current = self.score(np.exp(log_w), lam)
            w_norm = np.exp(log_w) / np.exp(log_w).sum()
            print(f"[{i+1:2d}] score={current:.4f}  weights={np.round(w_norm, 3)}")
            best_score = max(best_score, current)

        return best_score, np.exp(log_w) / np.exp(log_w).sum()


if __name__ == "__main__":
    dataset_train = KernelDataset(x_csv_path="./Xtr.csv", y_csv_path="./Ytr.csv")
    dataset_test  = KernelDataset(x_csv_path="./Xte.csv")

    good_configs = load_good_configs("kernel_results.json")
    if len(good_configs) < 2:
        print("Not enough good kernels for MKL — run test_regression.py first.")
        exit(1)

    # Build a temporary Trainer just to get the train/val split and kernel matrices
    # Use first config's kernel to initialize the split (splits are deterministic)
    first = good_configs[0]
    k0    = build_kernel(first["kernel"], **{**first["params"], **first["fixed"]})
    dl    = KernelDataLoader(dataset_train, k0, max_size=5000)
    base_trainer = Trainer(dl)
    base_trainer.fit()  # establishes the train/val split

    y_train = base_trainer.y_train
    y_val   = base_trainer.y_val
    assert y_train is not None and y_val is not None

    # Compute all kernel matrices using the same split
    print("\nComputing kernel matrices for all good configs...")
    Ks_train, Ks_val = [], []
    for r in good_configs:
        k = build_kernel(r["kernel"], **{**r["params"], **r["fixed"]})
        dl_sub = KernelDataLoader(dataset_train, k, max_size=5000)
        # Reuse the same split indices
        assert base_trainer.dl_train is not None
        dl_sub.dataset = base_trainer.dl_train.dataset  # same train subset
        K_train = dl_sub.get_kernel_matrix()
        K_val   = base_trainer.dl_train.get_kernel_matrix(
            Y=base_trainer.dataloader.split()[1].dataset.images.reshape(-1, 3072)
        )
        Ks_train.append(K_train)
        Ks_val.append(K_val)
        print(f"  {r['kernel']} done — K_train {K_train.shape}")

    # Optimize weights
    print("\nOptimizing MKL weights...")
    mkl = MKLTrainer(Ks_train, Ks_val, y_train, y_val)

    best_score = 0.0
    for lam in [100.0, 1000.0, 10000.0]:
        print(f"\n--- λ={lam} ---")
        score, weights = mkl.optimize_weights(lam=lam, n_iter=20, lr=0.3, eps=0.2)
        print(f"Best score: {score:.4f}  weights: {weights}")
        if score > best_score:
            best_score = score
            K_train, K_val = mkl.combined(weights)
            reg = Regularizer(lam=lam)
            strategy = OneVsAllStrategy(n_classes=10, model_factory=lambda: KernelRidgeRegression(reg))
            strategy.fit(K_train, y_train)
            # Predict on test
            assert base_trainer.dl_train is not None
            K_test = sum(weights[i] * base_trainer.dl_train.get_kernel_matrix(
                Y=dataset_test.images.reshape(-1, 3072)
            ) for i in range(len(weights)))
            preds = strategy.predict(K_test)
            prediction_to_csv(preds)
            print(f"*** New best {best_score:.4f} — saved ***")

    print(f"\nFinal MKL val accuracy: {best_score:.4f}")
