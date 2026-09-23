"""Tier-0 student embedding: a compact 2D representation for fast routing.

The student embedding is a PCA-compressed (via covariance eigendecomposition)
combination of folded ECFP counts and normalised physicochemical properties.
It requires no conformer generation or force-field work at query time, which
is what makes millisecond-scale Tier-0 routing over large libraries possible.

Roadmap: distil the multi-expert teacher score into this space (train the
projection so that student cosine similarity mimics the full cascade score)
once labelled pair data from campaigns accumulates.  The API is deliberately
shaped for that upgrade (``fit`` / ``transform`` / persistence).
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from aegis.chemistry import compute_properties, property_vector


class StudentEmbedder:
    """Folded-ECFP + property embedding with an optional learned projection."""

    def __init__(self, radius: int = 2, bits: int = 1024, dim: int = 256):
        self.radius = radius
        self.bits = bits
        self.dim = dim
        self._generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None

    def raw_features(self, mol: Chem.Mol) -> np.ndarray:
        """Length ``bits + 8`` raw feature vector for one molecule."""
        counts = np.zeros(self.bits, dtype=float)
        fingerprint = self._generator.GetCountFingerprint(mol)
        for index, value in fingerprint.GetNonzeroElements().items():
            counts[index] = float(value)
        total = counts.sum()
        if total > 0:
            counts = counts / total
        properties = property_vector(compute_properties(mol))
        return np.concatenate([counts, properties])

    def fit(self, mols: list[Chem.Mol], max_rows: int = 20000) -> "StudentEmbedder":
        """Learn the PCA projection from a (subsample of a) library."""
        if len(mols) < 2:
            return self
        if len(mols) > max_rows:
            rng = np.random.default_rng(0)
            indices = rng.choice(len(mols), size=max_rows, replace=False)
            mols = [mols[int(i)] for i in indices]
        matrix = np.vstack([self.raw_features(mol) for mol in mols])
        self.mean_ = matrix.mean(axis=0)
        centered = matrix - self.mean_
        covariance = centered.T @ centered / len(centered)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1][: self.dim]
        self.components_ = eigenvectors[:, order].T  # (dim, features)
        return self

    def transform(self, mol: Chem.Mol) -> np.ndarray:
        """Project one molecule into the (L2-normalised) student space."""
        features = self.raw_features(mol)
        if self.mean_ is not None and self.components_ is not None:
            features = (features - self.mean_) @ self.components_.T
        norm = float(np.linalg.norm(features))
        if norm > 0.0:
            features = features / norm
        return features

    def save(self, path: str) -> None:
        """Persist the projection to ``.npz``."""
        mean = self.mean_ if self.mean_ is not None else np.zeros(0)
        components = self.components_ if self.components_ is not None else np.zeros(0)
        np.savez(path, mean=mean, components=components)

    @classmethod
    def load(
        cls,
        path: str,
        radius: int = 2,
        bits: int = 1024,
        dim: int = 256,
    ) -> "StudentEmbedder":
        """Load a persisted projection."""
        embedder = cls(radius=radius, bits=bits, dim=dim)
        data = np.load(path)
        mean = data["mean"] if "mean" in data.files else np.zeros(0)
        components = data["components"] if "components" in data.files else np.zeros(0)
        embedder.mean_ = mean if mean.size else None
        embedder.components_ = components if components.size else None
        return embedder
