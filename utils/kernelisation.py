import jax.numpy as jnp
import pandas as pd
import numpy as np
from jax import vmap, jit
import jax.random as rd
from jaxtyping import Float, Array, Integer
from typing import Callable, List, Optional, Generic, TypeVar
from tqdm import tqdm
from sklearn.model_selection import train_test_split

A = TypeVar('A')
R = TypeVar('R')

class  KernelBase(Generic[A, R]):
    k: Callable[[A, A], R]

    def __call__(self, x: A, y: A) -> R:
        return self.k(x, y)

    def to_tensor(self, X: List[A], Y: Optional[List[A]])-> Float[Array, "N, M"]:
        del X, Y
        return jnp.ones((1, 1))

class Kernel( KernelBase[A, float]):
    def __init__(
        self,
        k: Callable[[A, A], float],
    ) -> None:
        self.k = k

    def to_tensor(
        self,
        X: List[A],
        Y: Optional[List[A]] = None
    ) -> Float[Array, "N M"]:
        if Y is None:
            Y = X
        return jnp.array([
            [self.k(e1, e2) for e1 in X]
            for e2 in tqdm(Y, desc="Building kernel matrix")
        ])


class JaxKernel( KernelBase[Float[Array, "N"], Float[Array, ""]]):
    """Kernel where k is always a pure JAX function returning an Array."""
    def __init__(
        self,
        k: Callable[[Float[Array, "N"], Float[Array, "N"]], Float[Array, "1"]]
    ) -> None:
        self.k = k

    def to_tensor(
            self,
            X: Float[Array, "N D"],
            Y: Optional[Float[Array, "M D"]] = None
        ) -> Float[Array, "N M"]:
            k_jit = jit(self.k)
            if Y is None:
                Y = X
            return vmap(lambda y: vmap(lambda x: k_jit(x, y))(X))(Y)

class NystromKernel(JaxKernel):
    def __init__(self, base_kernel: JaxKernel, n_landmarks: int = 500):
        self.base_kernel = base_kernel
        self.n_landmarks = n_landmarks
        self._landmarks: Optional[Float[Array, "M D"]] = None
        self._K_mm_inv_sqrt: Optional[Float[Array, "M M"]] = None

    def fit_landmarks(self, X: Float[Array, "N D"]) -> None:
        idx = rd.choice(rd.PRNGKey(0), X.shape[0], (self.n_landmarks,), replace=False)
        self._landmarks = X[idx]
        K_mm = self.base_kernel.to_tensor(self._landmarks)
        eigvals, eigvecs = jnp.linalg.eigh(K_mm)
        eigvals = jnp.maximum(eigvals, 1e-8)
        self._K_mm_inv_sqrt = eigvecs * (1.0 / jnp.sqrt(eigvals))

    def to_tensor(self, X: Float[Array, "N D"], Y: Optional[Float[Array, "M D"]] = None) -> Float[Array, "N M"]:
        if self._landmarks is None or self._K_mm_inv_sqrt is None:
            raise ValueError("Landmarks not fit yet")
        K_nm = self.base_kernel.to_tensor(X, self._landmarks).T
        Z_n = K_nm @ self._K_mm_inv_sqrt
        if Y is None:
            return Z_n @ Z_n.T
        K_ym = self.base_kernel.to_tensor(Y, self._landmarks).T
        Z_y = K_ym @ self._K_mm_inv_sqrt
        return Z_n @ Z_y.T


class KernelDataset:
    labels: Optional[Integer[Array, "N"]] = None

    def __init__(self, x_csv_path: str, y_csv_path: Optional[str] = None, normalize:bool=False):
        pixel_data = np.array(pd.read_csv(x_csv_path, header=None, sep=',', usecols=range(3072)))
        self.images: Float[Array, "N 3 32 32"] = jnp.array(pixel_data, dtype=jnp.float32).reshape(-1, 3, 32, 32)
        if normalize:
            images = self.images.reshape(-1, 3, 1024)
            mean = images.mean(axis=2, keepdims=True)
            std  = images.std(axis=2, keepdims=True) + 1e-8
            self.images = ((images - mean) / std).reshape(-1, 3, 32, 32)

        if y_csv_path is not None:
            labels = np.array(pd.read_csv(y_csv_path, sep=',', usecols=[1])).squeeze()
            self.labels = jnp.array(labels, dtype=jnp.int32)

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx: int):
        img = self.images[idx]
        return (img, self.labels[idx]) if self.labels is not None else (img, idx)


class KernelDataLoader:
    def __init__(self, dataset: KernelDataset, kernel: Optional[JaxKernel] = None, 
                 batch_size: int = 64, shuffle: bool = True, max_size: Optional[int] = None):
        if max_size is not None:
            sub = KernelDataset.__new__(KernelDataset)
            sub.images = dataset.images[:max_size]
            sub.labels = dataset.labels[:max_size] if dataset.labels is not None else None
            dataset = sub
        self.dataset = dataset
        self.kernel = kernel
        self.batch_size = batch_size
        self.shuffle = shuffle

    def fit_kernel(self) -> None:
        """Fit landmarks if kernel is Nystrom. No-op otherwise."""
        if isinstance(self.kernel, NystromKernel):
            images = self.dataset.images.reshape(-1, 3072)
            self.kernel.fit_landmarks(images)

    def split(self, test_size: float = 0.2, random_state: int = 42) -> tuple["KernelDataLoader", "KernelDataLoader"]:
        # Removed the NystromKernel guard — landmarks are fit after splitting now
        n = len(self.dataset)
        idx = jnp.arange(n)
        idx_train, idx_val = train_test_split(idx, test_size=test_size, random_state=random_state)
        idx_train, idx_val = jnp.array(idx_train), jnp.array(idx_val)

        def subset(idx: Integer[Array, "N"]) -> "KernelDataLoader":
            sub = KernelDataset.__new__(KernelDataset)
            sub.images = self.dataset.images[idx]
            sub.labels = self.dataset.labels[idx] if self.dataset.labels is not None else None
            return KernelDataLoader(sub, self.kernel, self.batch_size, self.shuffle, max_size=None)

        return subset(idx_train), subset(idx_val)


    def get_kernel_matrix(self, Y: Optional[Float[Array, "M D"]] = None) -> Float[Array, "N M"]:
        if self.kernel is None:
            raise ValueError("No kernel provided.")
        if isinstance(self.kernel, NystromKernel) and self.kernel._landmarks is None:
            raise ValueError(
                "NystromKernel landmarks not fitted. Call fit_kernel() before get_kernel_matrix()."
            )
        images = self.dataset.images.reshape(-1, 3072)
        return self.kernel.to_tensor(images, Y)

    def __iter__(self):
        n = len(self.dataset)
        indices = np.random.permutation(n) if self.shuffle else np.arange(n)
        for i in range(0, n, self.batch_size):
            batch_idx = indices[i:i + self.batch_size]
            imgs = self.dataset.images[batch_idx]
            if self.dataset.labels is not None:
                yield imgs, self.dataset.labels[batch_idx]
            else:
                yield imgs, batch_idx
