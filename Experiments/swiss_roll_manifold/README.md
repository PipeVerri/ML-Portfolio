## Swiss Roll Manifold & PCA

This directory contains experiments on dimensionality reduction applied to a synthetic manifold. The goal is to understand the limitations of linear methods (PCA) when the underlying structure is non-linearly embedded in higher-dimensional space.

### Notebooks

- `swiss_roll_pca.ipynb`: Analysis of PCA applied to a custom Swiss Roll dataset. The experiment is structured in three stages:
  - **Dataset Generation**: A linear function `y = 2x + 1` is used as the ground-truth signal, which is then embedded into 3D space using the Swiss Roll parametrization, with independent train and test sets.
  - **Manual PCA via Covariance Eigenvectors**: The covariance matrix of the centered training data is decomposed to obtain principal directions, which are visualized as scaled vectors over the 3D point cloud.
  - **Sklearn PCA Projection & Reconstruction**: A 2-component PCA is fit on the training data. The 2D projection (colored by the original `x` parameter) and the 3D reconstruction are compared to evaluate how much structure is preserved by the linear projection.

### Conclusions

PCA is expected to fail at recovering the true structure of the Swiss Roll. "Unrolling" the manifold requires a non-linear transformation — points that are far apart in Euclidean 3D space may be close along the surface of the roll, and vice versa. Since PCA operates purely through linear operations on the covariance matrix, it has no way to capture this intrinsic geometry. The projection onto principal components collapses the roll rather than unfolding it, mixing points that were well-separated along the original signal.

### Data

All data is synthetically generated — no external datasets are used. The Swiss Roll is constructed programmatically from a linear signal embedded via trigonometric parametrization.