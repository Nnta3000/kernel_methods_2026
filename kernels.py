import jax.numpy as jnp
from jaxtyping import Float, Array
import utils

class GaussianKernel(utils.Kernel):
    def __init__(self, sigma: float):
        k = lambda x, y: jnp.exp(-jnp.sum((x - y) ** 2) / (2 * sigma ** 2))
        super().__init__(k)
        self.is_jax = True

class LaplacianKernel(utils.Kernel):
    def __init__(self, sigma: float):
        k = lambda x, y: jnp.exp(-jnp.sum(jnp.abs(x - y)) / sigma)
        super().__init__(k)
        self.is_jax = True

class PolynomialKernel(utils.Kernel):
    def __init__(self, degree: int, c: float = 1.0):
        k = lambda x, y: (jnp.dot(x, y) / x.shape[0] + c) ** degree
        super().__init__(k)
        self.is_jax = True

class LinearKernel(utils.Kernel):
    def __init__(self):
        k = lambda x, y: jnp.dot(x,y)
        super().__init__(k)
        self.is_jax = True


class HistogramIntersectionKernel(utils.Kernel):
    def __init__(self):
        k = lambda x, y: jnp.sum(jnp.minimum(x, y))
        super().__init__(k)
        self.is_jax = True

class ArcCosineKernel(utils.Kernel):
    def __init__(self):
        def k(x, y):
            nx, ny = jnp.linalg.norm(x), jnp.linalg.norm(y)
            cos_theta = jnp.clip(jnp.dot(x, y) / (nx * ny + 1e-8), -1.0, 1.0)
            theta = jnp.arccos(cos_theta)
            return (1 / jnp.pi) * nx * ny * (jnp.sin(theta) + (jnp.pi - theta) * cos_theta)
        super().__init__(k)
        self.is_jax = True
