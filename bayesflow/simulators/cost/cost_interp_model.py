import os
import pathlib as pl
from collections.abc import Callable

import numpy as np

from bayesflow.utils import tree_stack, pickle_load


class CostInterpModel:
    def __init__(
        self,
        root: os.PathLike,
        *,
        pattern: str = "*.pkl",
        load_fn: Callable = None,
        cost_key: str = "cost",
    ):
        self.root = pl.Path(root)
        self.load_fn = load_fn or pickle_load
        self.cost_key = cost_key
        self.files = list(map(str, self.root.glob(pattern)))
        self.num_samples = len(self.files)

    def load(self) -> tuple[np.ndarray, np.ndarray]:
        batch = [self.load_fn(file) for file in self.files]
        batch = tree_stack(batch)
        theta = np.asarray(batch["theta"])
        cost = np.asarray(batch[self.cost_key])
        return theta, cost

    def fit(self, *, length_scale: float = 1.0, noise: float = 1e-6) -> "CostInterpModel":
        """
        Fit a Gaussian-process regressor with an RBF kernel to the data loaded by `load`.

        Parameters
        ----------
        length_scale : float, optional
            The length-scale of the RBF kernel. Default is 1.0.
        noise : float, optional
            The observation-noise variance added to the kernel diagonal. Default is 1e-6.

        Returns
        -------
        self : CostInterpModel
            The fitted model, to allow method chaining.
        """
        theta, cost = self.load()
        theta = np.asarray(theta, dtype=float)
        cost = np.asarray(cost, dtype=float)

        if theta.ndim == 1:
            theta = theta[:, np.newaxis]
        cost = cost.ravel()

        n = theta.shape[0]
        K = self._rbf_kernel(theta, theta, length_scale) + noise * np.eye(n)

        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, cost))

        self._theta = theta
        self._cost = cost
        self._L = L
        self._alpha = alpha
        self._length_scale = length_scale
        self._noise = noise
        return self

    def predict(self, theta_new) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict the mean and variance of the cost at new parameter values.

        Parameters
        ----------
        theta_new : array-like
            The new parameter values to predict at. Shape ``(m,)`` for a single parameter
            or ``(m, d)`` for `d` parameters.

        Returns
        -------
        mean : np.ndarray
            The predicted mean cost, with shape (m,).
        var : np.ndarray
            The predicted variance of the cost, with shape (m,).
        """
        if not hasattr(self, "_theta"):
            raise RuntimeError("Model not fitted. Call fit() first.")

        theta_new = np.asarray(theta_new, dtype=float)
        if theta_new.ndim == 1:
            theta_new = theta_new[:, np.newaxis]

        K_s = self._rbf_kernel(theta_new, self._theta, self._length_scale)
        mean = K_s @ self._alpha
        v = np.linalg.solve(self._L, K_s.T)
        var = self._length_scale**2 - np.sum(v**2, axis=1)
        return mean, var

    @staticmethod
    def _rbf_kernel(X1: np.ndarray, X2: np.ndarray, length_scale: float) -> np.ndarray:
        sq_dists = (
            np.sum(X1**2, axis=1)[:, None]
            + np.sum(X2**2, axis=1)[None, :]
            - 2.0 * X1 @ X2.T
        )
        sq_dists = np.maximum(sq_dists, 0.0)
        return np.exp(-0.5 * sq_dists / length_scale**2)

