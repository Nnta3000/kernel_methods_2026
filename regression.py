from dataclasses import dataclass
from typing import Protocol, Optional, Callable
import jax.numpy as jnp
from jax.nn import sigmoid
from jaxtyping import Float, Integer, Array
from tqdm import tqdm

@dataclass
class Regularizer:
    lam: float = 1.0

class RegressionModel(Protocol):
    def fit(self, K: Float[Array, "N N"], y: Float[Array, "N"]) -> None: ...
    def predict(self, K_test: Float[Array, "M N"]) -> Float[Array, "M"]: ...

class MulticlassStrategy(Protocol):
    def fit(self, K: Float[Array, "N N"], y: Integer[Array, "N"]) -> None: ...
    def predict(self, K_test: Float[Array, "M N"]) -> Integer[Array, "M"]: ...


class KernelRidgeRegression(RegressionModel):
    def __init__(self, regularizer: Regularizer):
        self.lam = regularizer.lam
        self.alpha: Optional[Float[Array, "N C"]] = None

    def fit(self, K: Float[Array, "N N"], y: Float[Array, "N C"]) -> None:
        n = K.shape[0]
        self.alpha = jnp.linalg.solve(K + self.lam * n * jnp.eye(n), y)

    def predict(self, K_test: Float[Array, "M N"]) -> Float[Array, "M C"]:
        if self.alpha is None:
            raise ValueError("Model not fitted yet.")
        return K_test @ self.alpha


class Classifier:
    def __init__(self, strategy: MulticlassStrategy):
        self.strategy = strategy

    def fit(self, K: Float[Array, "N N"], y: Integer[Array, "N"]) -> None:
        self.strategy.fit(K, y)

    def predict(self, K_test: Float[Array, "M N"]) -> Integer[Array, "M"]:
        return self.strategy.predict(K_test)

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

class KernelLogisticRegression(RegressionModel):
    def __init__(self, regularizer: Regularizer, n_iter: int = 10):
        self.lam = regularizer.lam
        self.n_iter = n_iter
        self.alpha: Optional[Float[Array, "N"]] = None


    def fit(self, K: Float[Array, "N N"], y: Float[Array, "N"]) -> None:
        n = K.shape[0]
        alpha = jnp.zeros(n)
        for _ in range(self.n_iter):
            m = K @ alpha
            W = sigmoid(m) * sigmoid(-m)
            z = m + y * sigmoid(-y * m) / W
            A = K + n * self.lam * jnp.diag(1.0 / W)
            alpha = jnp.linalg.solve(A, z)

        self.alpha = alpha

    def predict(self, K_test: Float[Array, "M N"]) -> Float[Array, "M"]:
        if self.alpha is None:
            raise ValueError("Model not fitted yet.")
        return K_test @ self.alpha  # return raw scores, let OneVsAll do argmax

