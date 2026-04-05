import numpy as np


class PCAScratch:
    """Principal Component Analysis implemented from scratch.

    Parameters
    ----------
    n_components : int or None, default=None
        Number of principal components to keep. If None, all components are kept.
    method : {'svd', 'eigen'}, default='svd'
        Computation method used to obtain principal components.
        - 'svd': compute PCA via singular value decomposition of the centered data.
        - 'eigen': compute PCA via eigen decomposition of the covariance matrix.
    whiten : bool, default=False
        If True, the components are scaled to have unit variance.
    """

    def __init__(self, n_components=None, method='svd', whiten=False):
        self.n_components = n_components
        self.method = method
        self.whiten = whiten

        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
        self.n_features_ = None
        self.n_samples_ = None

    def fit(self, X, y=None):
        """Fit the model with X."""
        X = self._validate_input(X)
        self.n_samples_, self.n_features_ = X.shape
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        n_components = self.n_components
        if n_components is None:
            n_components = min(self.n_samples_, self.n_features_)
        if not 1 <= n_components <= min(self.n_samples_, self.n_features_):
            raise ValueError(
                "n_components must be between 1 and min(n_samples, n_features)"
            )

        method = self.method.lower()
        if method == 'svd':
            U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
            components = Vt[:n_components]
            explained_variance = (S[:n_components] ** 2) / (self.n_samples_ - 1)
            total_var = (S ** 2).sum() / (self.n_samples_ - 1)
            singular_values = S[:n_components]
        elif method == 'eigen':
            covariance = np.dot(X_centered.T, X_centered) / (self.n_samples_ - 1)
            eigvals, eigvecs = np.linalg.eigh(covariance)
            sorting = np.argsort(eigvals)[::-1]
            eigvals = eigvals[sorting]
            eigvecs = eigvecs[:, sorting]
            components = eigvecs[:, :n_components].T
            explained_variance = eigvals[:n_components]
            total_var = eigvals.sum()
            singular_values = np.sqrt(explained_variance * (self.n_samples_ - 1))
        else:
            raise ValueError("method must be 'svd' or 'eigen'")

        self.components_ = components
        self.explained_variance_ = explained_variance
        self.explained_variance_ratio_ = explained_variance / total_var
        self.singular_values_ = singular_values
        self.n_components_ = n_components
        return self

    def transform(self, X):
        """Apply dimensionality reduction to X."""
        if self.components_ is None or self.mean_ is None:
            raise ValueError("The PCA model has not been fitted yet.")
        X = self._validate_input(X)
        X_centered = X - self.mean_
        X_transformed = np.dot(X_centered, self.components_.T)
        if self.whiten:
            scale = np.sqrt(self.explained_variance_)
            X_transformed = X_transformed / scale
        return X_transformed

    def fit_transform(self, X, y=None):
        """Fit the model with X and apply the dimensionality reduction on X."""
        self.fit(X, y=y)
        return self.transform(X)

    def inverse_transform(self, X):
        """Transform data back to its original space."""
        if self.components_ is None or self.mean_ is None:
            raise ValueError("The PCA model has not been fitted yet.")
        if X.ndim != 2 or X.shape[1] != self.n_components_:
            raise ValueError(
                f"X must have shape (n_samples, {self.n_components_})"
            )
        X_original = np.dot(X, self.components_) + self.mean_
        return X_original

    def _validate_input(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("Input data must be a 2D array.")
        if X.size == 0:
            raise ValueError("Input data must not be empty.")
        return X


class PCASVD(PCAScratch):
    """PCA using singular value decomposition."""

    def __init__(self, n_components=None, whiten=False):
        super().__init__(n_components=n_components, method='svd', whiten=whiten)


class PCAEigen(PCAScratch):
    """PCA using eigen decomposition of the covariance matrix."""

    def __init__(self, n_components=None, whiten=False):
        super().__init__(n_components=n_components, method='eigen', whiten=whiten)


__all__ = ['PCAScratch', 'PCASVD', 'PCAEigen']
