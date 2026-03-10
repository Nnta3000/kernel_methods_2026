import torch 
import pandas as pd
import numpy as np
from typing import Callable, List, Optional, Protocol, TypeVar

A = TypeVar('A')

class Kernel(Protocol[A]):
    def _null_kernel(_1:A,_2:A)->float:
        del _1,_2
        return 0
    k: Callable[[A, A], float]= _null_kernel
    def __init__(self, k:Callable[[A, A], float], device ) -> None:
        self.k = k
    def __call__(self, x, y)-> float:
        return self.k(x, y)
    def to_tensor(self, elements: List[A] ) -> torch.Tensor:
        return torch.Tensor([[self.k(e1,e2) for e1 in elements] for e2 in elements])

class KaggleDataset(torch.utils.data.Dataset):
    labels:Optional[torch.Tensor] = None
    def __init__(self, x_csv_path: str, y_csv_path: Optional[str] = None):
        pixel_data = np.array(pd.read_csv(x_csv_path, header=None, sep=',', usecols=range(3072)))
        self.images = torch.tensor(pixel_data, dtype=torch.float32).view(-1, 3, 32, 32)

        if y_csv_path is not None:
            labels = np.array(pd.read_csv(y_csv_path, sep=',', usecols=[1])).squeeze()
            self.labels = torch.tensor(labels, dtype=torch.long)
        else:
            self.labels = None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        return (img, self.labels[idx]) if self.labels is not None else (img, idx)

class GaussianKernel(Kernel):
    def __init__(self, sigma: float, device='cpu'):
        k = lambda x, y: torch.exp(-torch.sum((x - y) ** 2) / (2 * sigma ** 2)).item()
        super().__init__(k, device)
