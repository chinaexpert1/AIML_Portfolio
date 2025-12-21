
import numpy as np
from typing import Iterable, Optional, Tuple

class Embedding:
    """
    Minimal learnable embedding layer implemented with NumPy.
    - num_embeddings: vocabulary size (V)
    - embedding_dim: feature size (D)

    API:
      forward(indices) -> (N, D) array of selected rows
      zero_grad() -> clears internal gradient buffer
      backward(indices, grad_output) -> accumulates dLoss/dW for used rows
      step(lr) -> SGD update on weights using accumulated grads
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, seed: Optional[int] = 42):
        rng = np.random.default_rng(seed)
        self.weight = rng.normal(loc=0.0, scale=0.02, size=(num_embeddings, embedding_dim)).astype(np.float64)
        self.grad = np.zeros_like(self.weight)

    def forward(self, indices: Iterable[int]) -> np.ndarray:
        idx = np.fromiter(indices, dtype=np.int64)
        return self.weight[idx]

    def zero_grad(self):
        self.grad.fill(0.0)

    def backward(self, indices: Iterable[int], grad_output: np.ndarray):
        """
        Accumulate gradients wrt weight rows:
          grad_output: shape (N, D), where N == len(indices)
        """
        idx = np.fromiter(indices, dtype=np.int64)
        if grad_output.shape != (idx.shape[0], self.weight.shape[1]):
            raise ValueError(f"grad_output shape {grad_output.shape} does not match (N, D)")
        # Efficient scatter-add into rows
        # If indices repeat, accumulate their gradients
        for i, row_id in enumerate(idx):
            self.grad[row_id] += grad_output[i]

    def step(self, lr: float = 1e-2):
        self.weight -= lr * self.grad
        self.zero_grad()
