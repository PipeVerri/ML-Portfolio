# Machine Learning Portfolio

A collection of experiments and projects focused on computer vision, dimensionality reduction, and unsupervised learning, using both standard frameworks and from-scratch implementations.

## Experiments

This section contains in-depth machine learning analyses, documented in Jupyter Notebooks.

- **[MNIST Sparse Contractive Autoencoder](./Experiments/AE_MNIST)**:
  * Exploration of a 4-dimensional latent space trained to behave like discrete binary “switches” using sparsity (L1) and contractive regularization.
  * Interactive analysis tools for inspecting reconstructions and manually manipulating latent bits to observe their semantic effects on generated digits.

- **[Object Detection via Clustering](./Experiments/object_detection_clustering/)**: 
  - Unsupervised object detection pipeline on PASCAL VOC 2007.
  - Feature extraction using Sobel convolutions and spatial clustering (K-Means, DBSCAN, OPTICS, Birch).
  - Hyperparameter optimization via a concurrent grid search scored by a pretrained CNN ensemble (MobileNetV3, SqueezeNet).

- **[Dog vs. Cat Classifier & Loss Landscapes](./Experiments/dog_cat_classifier/)**: 
  - Binary image classification using custom and compact CNN architectures.
  - Exploration of the loss landscape geometry via random-direction perturbation analysis and 3D surface visualization.
  - Preliminary feature inspection using fixed Sobel kernels.

- **[Manifold Learning & PCA](./Experiments/swiss_roll_manifold/)**: 
  - Comparative analysis of PCA on non-linear structures using the Swiss Roll manifold.
  - Implementation of PCA via covariance eigendecomposition and reconstruction fidelity assessment.
  - Study of the limitations of linear dimensionality reduction on non-linear embeddings.

- **[Image Classification via PCA](./Experiments/dog_cat_pca/)**: 
  - Large-scale image classification using PCA for dimensionality reduction on the Dogs vs. Cats dataset.
  - Tractable eigendecomposition using the Gram Matrix method ($XX^T$).
  - Benchmark between custom SGD Logistic Regression and scikit-learn implementations.

- **[Unsupervised Clustering (K-Means)](./Experiments/k_means/)**: 
  - From-scratch K-Means implementation applied to the Fashion MNIST dataset.
  - Cluster purity analysis and evaluation of semantic grouping based on raw pixel intensity.

- **[XOR Classification with Polynomial SVM](./Experiments/svm_test/)**: 
  - Investigation of the `coef0` hyperparameter in polynomial kernels for non-linearly separable data.
  - Visualization of decision boundaries on canonical XOR and Gaussian blob patterns.

## Project Structure

```text
├── assets/             # Visualization artifacts and plots
├── Experiments/        # Research notebooks and experiment documentation
├── pyproject.toml      # Project dependencies and metadata
└── README.md           # Portfolio overview
```
