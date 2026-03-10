import jax.numpy as jnp
import pandas as pd
import numpy as np
from jax import vmap, jit
from jaxtyping import Float, Array, Integer
from typing import Callable, List, Optional, Protocol, TypeVar
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

A = TypeVar('A')




class Kernel(Protocol[A]):
    def _null_kernel(_1: A, _2: A) -> float:
        del _1, _2
        return 0
    k: Callable[[A, A], float] = _null_kernel
    is_jax: bool=False
    def __init__(self, k: Callable[[A, A], float], device='cpu', is_jax:bool=False) -> None:
        self.k = k
        self.is_jax = is_jax
    def __call__(self, x: A, y: A) -> float:
        return self.k(x, y)
    def to_tensor(self, X: List[A], Y: Optional[List[A]] = None) -> Float[Array, "N M"]:
        if getattr(self, 'is_jax', False):
            X_arr = jnp.array(X)
            Y_arr = jnp.array(Y) if Y is not None else X_arr
            return self.to_tensor_jax(X_arr, Y_arr)
        if Y is None:
            Y = X
        return jnp.array([[self.k(e1, e2) for e1 in X] for e2 in tqdm(Y, desc="Building kernel matrix")])
    def to_tensor_jax(self, X: Float[Array, "N D"], Y: Optional[Float[Array, "M D"]] = None) -> Float[Array, "N M"]:
        """Should be call if the kernel is a pure JAX function"""
        try:
            k_jit = jit(self.k)
            k_jit(X[0], X[0])
        except Exception as e:
            raise ValueError(f"self.k does not appear to be a pure JAX function: {e}")
        if Y is None:
            Y = X
        return vmap(lambda y: vmap(lambda x: k_jit(x, y))(X))(Y)


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
    def __init__(self, dataset: KernelDataset, kernel: Optional[Kernel] = None, 
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

    def split(self, test_size: float = 0.2, random_state: int = 42) -> tuple["KernelDataLoader", "KernelDataLoader"]:
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
        images = self.dataset.images.reshape(-1, 3072)
        X = list(images)
        Y_list = list(Y) if Y is not None else None
        return self.kernel.to_tensor(X, Y_list)

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
