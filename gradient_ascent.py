from typing import Any, Callable
import gc, jax, json
import numpy as np
import kernels as kernel_module, classifiers
from utils.classification import OneVsAllStrategy, Trainer
from utils.kernelisation import KernelDataset, KernelDataLoader
from utils.tools import prediction_to_csv
from dataclasses import dataclass, field


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class KernelConfig:
    name: str
    algo: str
    init_params: dict[str, float]   # continuous params optimized in log space
    fixed_params: dict[str, Any] = field(default_factory=dict)


def build_kernel(name: str, **params):
    return {
        "Anova":     lambda: kernel_module.AnovaKernel(sigma=params["sigma"], p=2),
        "Gaussian":  lambda: kernel_module.GaussianKernel(sigma=params["sigma"]),
        "Laplacian": lambda: kernel_module.LaplacianKernel(sigma=params["sigma"]),
        "Poly2":     lambda: kernel_module.PolynomialKernel(degree=2),
        "Poly3":     lambda: kernel_module.PolynomialKernel(degree=3),
        "Histogram": lambda: kernel_module.HistogramIntersectionKernel(),
        "Linear":    lambda: kernel_module.LinearKernel(),
    }[name]()


# ── Objective ─────────────────────────────────────────────────────────────────

MAX_CACHE_SIZE = 3
RANDOM_ACC     = 0.10

# Full plausible range for each param in log space
LOG_RANGES = {
    "lam":   (np.log(1e-2), np.log(1e4)),
    "sigma": (np.log(1e-1),  np.log(1e4)),
}


def make_objective(
    dataset_train: KernelDataset,
    dataset_test:  KernelDataset,
    best_score_ref: list[float],
) -> Callable[..., float]:

    cache: dict[str, Trainer] = {}

    def objective(config: KernelConfig, **params: float) -> float:
        all_params = {**params, **config.fixed_params}
        cache_key  = f"{config.name}_" + "_".join(
            f"{k}={v:.4f}" for k, v in sorted(all_params.items()) if k != "lam"
        )

        if cache_key not in cache:
            if len(cache) >= MAX_CACHE_SIZE:
                del cache[next(iter(cache))]
                gc.collect()
                jax.clear_caches()
            print(f"  [Cache miss] {cache_key}")
            dl = KernelDataLoader(dataset_train, build_kernel(config.name, **all_params), max_size=5000)
            trainer = Trainer(dl)
            trainer.fit()
            cache[cache_key] = trainer

        trainer = cache[cache_key]
        reg     = classifiers.Regularizer(lam=params["lam"])
        factory = {
            "KRR": lambda: classifiers.KernelRidgeRegression(reg),
            "KLR": lambda: classifiers.KernelLogisticRegression(reg, n_iter=20),
            "SVM": lambda: classifiers.KernelSVM(reg),
        }[config.algo]

        trainer.strategy = OneVsAllStrategy(n_classes=10, model_factory=factory)
        trainer.refit()
        train_acc, val_acc = trainer.score()
        gap = train_acc - val_acc

        # Composite score:
        #   - underfit (train ~ random): heavy penalty
        #   - overfit  (high train, large gap): penalize gap
        #   - in between: just val_acc
        if train_acc < RANDOM_ACC + 0.05:
            composite = val_acc - 2.0
        elif train_acc > 0.7 and gap > 0.3:
            composite = val_acc - 0.5 * gap
        else:
            composite = val_acc

        print(f"    train={train_acc:.4f}  val={val_acc:.4f}  gap={gap:.4f}  composite={composite:.4f}")

        if val_acc > best_score_ref[0]:
            best_score_ref[0] = val_acc
            prediction_to_csv(trainer.predict(dataset_test))
            print(f"  *** New best val={val_acc:.4f} ({config.algo}, {config.name}, {params}) — saved ***")

        return composite

    return objective


# ── Gradient Ascent ───────────────────────────────────────────────────────────

class HyperparamGradientAscent:
    def __init__(self, objective: Callable, lr: float = 0.3, n_iter: int = 20, eps: float = 0.2, n_restarts: int = 4):
        self.objective  = objective
        self.lr         = lr
        self.n_iter     = n_iter
        self.eps        = eps
        self.n_restarts = n_restarts

    def _run_single(self, config: KernelConfig, log_params: dict[str, float]) -> tuple[float, dict[str, float]]:
        def score(**lp):
            return self.objective(config, **{k: float(np.exp(v)) for k, v in lp.items()})

        best_score  = score(**log_params)
        best_params = dict(log_params)

        for i in range(self.n_iter):
            eps = self.eps
            for _ in range(5):
                grads = {
                    key: (score(**{**log_params, key: log_params[key] + eps}) -
                          score(**{**log_params, key: log_params[key] - eps})) / (2 * eps)
                    for key in log_params
                }
                if any(abs(g) > 1e-6 for g in grads.values()):
                    break
                eps *= 2.0
                print(f"  [Null gradient] eps → {eps:.2f}")

            for key in log_params:
                log_params[key] += self.lr * grads[key]

            current = score(**log_params)
            print(f"  [{i+1:2d}] score={current:.4f}  " +
                  "  ".join(f"{k}={np.exp(v):.2f}" for k, v in log_params.items()))

            if current > best_score:
                best_score  = current
                best_params = dict(log_params)

        return best_score, best_params

    def run(self, config: KernelConfig) -> tuple[float, dict[str, float]]:
        best_score_global  = -np.inf
        best_params_global = {}

        for restart in range(self.n_restarts):
            if restart == 0:
                # First restart: use provided init
                log_params = {k: np.log(v) for k, v in config.init_params.items()}
            else:
                # Subsequent restarts: sample uniformly over full log range
                log_params = {
                    k: np.random.uniform(*LOG_RANGES[k])
                    for k in config.init_params
                }

            print(f"\n[Restart {restart+1}/{self.n_restarts}]  init={ {k: f'{np.exp(v):.2f}' for k,v in log_params.items()} }")
            score, params = self._run_single(config, log_params)

            if score > best_score_global:
                best_score_global  = score
                best_params_global = params

        return best_score_global, {k: float(np.exp(v)) for k, v in best_params_global.items()}


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    dataset_train = KernelDataset(x_csv_path="./Xtr.csv", y_csv_path="./Ytr.csv")
    dataset_test  = KernelDataset(x_csv_path="./Xte.csv")
    best_score_ref = [0.38]
    results: list[dict] = []

    objective = make_objective(dataset_train, dataset_test, best_score_ref)
    search    = HyperparamGradientAscent(objective=objective, lr=0.3, n_iter=10, eps=0.2, n_restarts=4)

    configs = [
        KernelConfig("Anova",     "KRR", {"lam": 1000.0, "sigma": 2000.0}),
        KernelConfig("Poly2",     "KRR", {"lam": 1.0}),
        KernelConfig("Poly3",     "KRR", {"lam": 1.0}),
        KernelConfig("Histogram", "KRR", {"lam": 1.0}),
        KernelConfig("Linear",    "KRR", {"lam": 1.0}),
    ]

    for config in configs:
        print(f"\n{'='*60}\n{config.algo} + {config.name}\n{'='*60}")
        score, best_params = search.run(config)
        results.append({
            "kernel": config.name,
            "algo":   config.algo,
            "score":  score,
            "params": best_params,
            "fixed":  config.fixed_params,
        })
        with open("kernel_results.json", "w") as f:
            json.dump(results, f, indent=2)
        gc.collect()
        jax.clear_caches()

    print(f"\nFinal best val accuracy: {best_score_ref[0]:.4f}")
    print("Results saved to kernel_results.json")
