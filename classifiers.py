from typing import Optional 
import jax.numpy as jnp
from jax.nn import sigmoid
from jaxtyping import Float, Array
from utils.classification import RegressionModel, Regularizer


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

