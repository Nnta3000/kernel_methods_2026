from dataclasses import dataclass
from typing import Protocol, Callable, Optional
import jax.numpy as jnp
from jaxtyping import Float, Array, Integer
from tqdm import tqdm
from utils.kernelisation import KernelDataset, KernelDataLoader

@dataclass
class Regularizer:
    lam: float = 1.0

class RegressionModel(Protocol):
    def fit(self, K: Float[Array, "N N"], y: Float[Array, "N"]) -> None: ...
    def predict(self, K_test: Float[Array, "M N"]) -> Float[Array, "M"]: ...

class MulticlassStrategy(Protocol):
    def fit(self, K: Float[Array, "N N"], y: Integer[Array, "N"]) -> None: ...
    def predict(self, K_test: Float[Array, "M N"]) -> Integer[Array, "M"]: ...

class OneVsAllStrategy:
    def __init__(self, n_classes: int, model_factory: Callable[[], RegressionModel]):
        self.n_classes = n_classes
        self.models = [model_factory() for _ in range(n_classes)]
    def fit(self, K: Float[Array, "N N"], y: Integer[Array, "N"]) -> None:
        for c, model in enumerate(tqdm(self.models, desc="OneVsAll fitting")):
            y_binary = jnp.where(y == c, 1.0, -1.0)
            model.fit(K, y_binary)

    def predict(self, K_test: Float[Array, "M N"]) -> Integer[Array, "M"]:
        scores = jnp.stack([m.predict(K_test) for m in self.models], axis=1)
        return jnp.argmax(scores, axis=1)

class Classifier:
    def __init__(self, dataloader: KernelDataLoader):
        self.dataloader = dataloader
        self.strategy: Optional[MulticlassStrategy] = None
        self.K_train: Optional[Float[Array, "N N"]] = None
        self.K_val: Optional[Float[Array, "M N"]] = None
        self.y_train: Optional[Integer[Array, "N"]] = None
        self.y_val: Optional[Integer[Array, "M"]] = None
        self.dl_train: Optional[KernelDataLoader] = None

    def fit(self, test_size: float = 0.2) -> None:
        """Computes and caches kernel matrices and labels."""
        self.dl_train, dl_val = self.dataloader.split(test_size=test_size)
        self.K_train = self.dl_train.get_kernel_matrix()
        self.K_val = self.dl_train.get_kernel_matrix(Y=dl_val.dataset.images.reshape(-1, 3072))
        self.y_train = self.dl_train.dataset.labels
        self.y_val = dl_val.dataset.labels

    def score(self) -> tuple[float, float]:
        """Returns (train_acc, val_acc) without refitting."""
        assert self.strategy is not None and self.K_train is not None
        assert self.y_train is not None and self.y_val is not None
        train_acc = float(jnp.mean(self.strategy.predict(self.K_train) == self.y_train))
        assert self.K_val is not None
        val_acc = float(jnp.mean(self.strategy.predict(self.K_val) == self.y_val))
        print(f"train: {train_acc:.4f}, val: {val_acc:.4f}")
        return train_acc, val_acc

    def refit(self) -> tuple[float, float]:
        assert self.K_train is not None, "Call fit() first."
        assert self.strategy is not None, "Set a strategy first."
        assert self.y_train is not None
        self.strategy.fit(self.K_train, self.y_train)
        return self.score()

    def predict(self, dataset: KernelDataset) -> Integer[Array, "M"]:
        assert self.dl_train is not None and self.strategy is not None
        K_test = self.dl_train.get_kernel_matrix(Y=dataset.images.reshape(-1, 3072)).T
        return self.strategy.predict(K_test)


