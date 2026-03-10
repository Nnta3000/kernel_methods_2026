from typing import Optional 
import jax.numpy as jnp
from jax.nn import sigmoid
from jaxtyping import Float, Array
from utils.classification import RegressionModel, Regularizer
import osqp
import scipy.sparse as sp
import numpy as np

class KernelSVM:
    def __init__(self, regularizer: Regularizer, n_iter:int=100):
        self.lam = regularizer.lam
        self.alpha: Optional[Float[Array, "N"]] = None
        self.n_iter = n_iter

    def fit(self, K: Float[Array, "N N"], y: Float[Array, "N"]) -> None:
        n = K.shape[0]
        y_np = np.array(y, dtype=np.float64)
        K_np = np.array(K, dtype=np.float64)

        yKy = np.diag(y_np) @ K_np @ np.diag(y_np)
        P = sp.csc_matrix((1 / (2 * self.lam)) * yKy + 1e-8 * np.eye(n))
        q = -np.ones(n, dtype=np.float64)
        G = sp.csc_matrix(np.eye(n))
        l = np.zeros(n, dtype=np.float64)
        u = np.full(n, 1.0 / n, dtype=np.float64)

        solver = osqp.OSQP()
        solver.setup(P, q, G, l, u, warm_starting=True, verbose=False)
        sol = solver.solve()

        mu = np.array(sol.x)
        self.alpha = jnp.array(y_np * mu / (2 * self.lam))

    def predict(self, K_test: Float[Array, "M N"]) -> Float[Array, "M"]:
        if self.alpha is None:
            raise ValueError("Model not fitted yet.")
        return K_test @ self.alpha    

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

