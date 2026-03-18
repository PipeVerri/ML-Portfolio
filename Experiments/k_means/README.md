# Fashion MNIST Clustering — K-Means

This directory contains an unsupervised learning experiment applying K-Means clustering to the Fashion MNIST dataset, with a focus on evaluating the algorithm's ability to recover semantically meaningful groupings from raw pixel data.

## Notebooks

- `kmeans.ipynb`: End-to-end K-Means clustering pipeline on Fashion MNIST. It covers:
  - **Data loading & preprocessing**: Dataset retrieval via `kagglehub`, per-feature z-score normalization, and shuffling.
  - **Manual K-Means implementation**: A from-scratch implementation (K = 9, up to 100 iterations) using Euclidean distance and early stopping on centroid convergence.
  - **Cluster visualization & purity analysis**: For each cluster, a sample of assigned test images is displayed alongside train/test purity scores, defined as the fraction of samples belonging to the dominant class within the cluster.

## Data

The analysis uses the [Fashion MNIST](https://www.kaggle.com/datasets/zalando-research/fashionmnist) dataset (Zalando Research), consisting of 70,000 grayscale 28×28 images across 10 clothing categories, sourced via the `kagglehub` API.

## Findings

K-Means achieves high purity on categories with distinctive silhouettes that are geometrically dissimilar from the rest — such as trousers, bags, or ankle boots. Performance degrades on visually similar classes, where fine-grained texture and detail carry most of the discriminative information:

- **Coat vs. Shirt**: Both share a similar rectangular torso silhouette, making Euclidean distance in pixel space insufficient for reliable separation.
- **Sandal vs. Sneaker**: Structurally similar footwear shapes lead to frequent cross-cluster assignment.

This highlights a fundamental limitation of K-Means on raw pixel features: the algorithm captures coarse shape similarity but is insensitive to the higher-frequency patterns that distinguish visually related classes.