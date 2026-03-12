import jax.numpy as jnp
from jaxtyping import Array
from utils.kernelisation import JaxKernel

class AnovaKernel(JaxKernel):
    def __init__(self, sigma: float = 1.0, p: int = 1):
        self.sigma = sigma
        self.p = p
        if p == 1:
            def k(x, y):
                # Sum of scalar Gaussian kernels per dimension
                return jnp.sum(jnp.exp(-sigma * (x - y) ** 2))
            super().__init__(k)
        elif p == 2:
            def k(x, y):
                # p=1 term
                k1 = jnp.exp(-sigma * (x - y) ** 2)  # (D,)
                # p=2: all pairs (i,j), i<j — use outer product trick
                # Σ_{i<j} k1[i]*k1[j] = (Σk1)² - Σk1²) / 2
                s1 = jnp.sum(k1)
                s2 = jnp.sum(k1 ** 2)
                return (s1 ** 2 - s2) / 2.0
            super().__init__(k)
        else:
            raise ValueError("p > 2 is computationally intractable for D=3072")

class GaussianKernel(JaxKernel):
    def __init__(self, sigma: float):
        k = lambda x, y: jnp.exp(-jnp.sum((x - y) ** 2) / (2 * sigma ** 2))
        super().__init__(k)

class LaplacianKernel(JaxKernel):
    def __init__(self, sigma: float):
        k = lambda x, y: jnp.exp(-jnp.sum(jnp.abs(x - y)) / sigma)
        super().__init__(k)

class PolynomialKernel(JaxKernel):
    def __init__(self, degree: int, c: float = 1.0):
        k = lambda x, y: (jnp.dot(x, y) / x.shape[0] + c) ** degree
        super().__init__(k)

class LinearKernel(JaxKernel):
    def __init__(self):
        k = lambda x, y: jnp.dot(x, y)
        super().__init__(k)

class HistogramIntersectionKernel(JaxKernel):
    def __init__(self):
        k = lambda x, y: jnp.sum(jnp.minimum(jnp.abs(x), jnp.abs(y)))
        super().__init__(k)

class ArcCosineKernel(JaxKernel):
    def __init__(self):
        def k(x, y):
            nx, ny = jnp.linalg.norm(x), jnp.linalg.norm(y)
            cos_theta = jnp.clip(jnp.dot(x, y) / (nx * ny + 1e-8), -1.0, 1.0)
            theta = jnp.arccos(cos_theta)
            return (1 / jnp.pi) * nx * ny * (jnp.sin(theta) + (jnp.pi - theta) * cos_theta)
        super().__init__(k)
