## MNIST Sparse Contractive Autoencoder
**Live demo**: An interactive deployment is available at [fverri/SparseCAE-MNIST](https://huggingface.co/spaces/fverri/SparseCAE-MNIST) (Hugging Face Spaces). Use the four sliders to set each latent bit manually and observe the decoded digit in real time.

This directory contains an experiment in representation learning applied to the MNIST digit dataset. The goal is not to maximize reconstruction performance, but to experiment with autoencoders and push them toward a representation that is easy to manipulate by hand — where each latent dimension behaves like a switch that can be turned on or off.

### Notebooks

- `AE_MNIST.ipynb`: Training and analysis of a Sparse Contractive Autoencoder (SparseCAE) on MNIST. The experiment is structured in four stages:
  - **Data Loading & Preprocessing**: MNIST is loaded from raw binary files and standardized via `StandardScaler` fitted on the training split.
  - **Model Architecture & Training**: A fully-connected autoencoder with a 4-dimensional bottleneck is trained using PyTorch Lightning. The encoder bottleneck applies `ReLU → Tanh` to constrain activations to `[0, 1]`, and training is regularized with both a sparsity penalty and a contractive penalty (see *Model* below).
  - **Reconstruction Visualization**: Interactive widget to inspect original vs. reconstructed images alongside the 4-dimensional latent activation bar chart for any training or test sample.
  - **Latent Space Exploration**: Interactive widget with four sliders (`z[0]`–`z[3]`), each in `[0, 1]`, for manually setting the latent code and observing the decoded output in real time.

### Model

The `SparseCAE` is defined in `SparseCAE.py` and extends `pl.LightningModule`. Its design combines three losses:

- **Reconstruction** (`MSE`): standard pixel-level fidelity.
- **Sparsity** (`λ_sparse · ||z||₁`): L1 penalty on the latent activations. This discourages the model from using all dimensions simultaneously, pushing most "bits" toward zero.
- **Contractive** (`λ_cae · ||∂z/∂x||²_F`): Frobenius norm of the encoder Jacobian, computed via autograd. This penalizes sensitivity of the latent code to input perturbations. When combined with the `ReLU → Tanh` bottleneck, the contractive term pushes each activation toward the saturation regions of `Tanh` (near 0 or 1), where the derivative is small — effectively discouraging intermediate values.

The interaction between the two regularizers is the core inductive bias of this experiment: sparsity determines *how many* bits fire, and contractiveness determines *how cleanly* they saturate.

### Results

<!-- Insert reconstruction grid (original vs. reconstructed) here -->

<!-- Insert latent activation heatmap by digit class here -->

<!-- Insert examples of manual bit manipulation from the slider widget here -->

### Conclusions

The `ReLU → Tanh` bottleneck combined with sparsity and contractive regularization was designed to produce near-binary latent codes without explicitly discretizing them during training. The sparsity loss suppresses unused dimensions, while the contractive loss pushes active ones to saturate — together acting as a soft binarization. With only 4 latent dimensions, the model is forced to find a compact combinatorial structure over the 10 digit classes, which the slider widget makes directly interpretable.

### Data

MNIST is downloaded via `kagglehub`. No other external data is used. All preprocessing (flattening and standardization) is done in-notebook and is not persisted to disk.## MNIST Sparse Contractive Autoencoder

This directory contains an experiment in representation learning applied to the MNIST digit dataset. The goal is to train an autoencoder whose latent space behaves as a set of discrete binary "bits" rather than continuous values, enabling manual exploration of the learned encoding.

### Notebooks

- `AE_MNIST.ipynb`: Training and analysis of a Sparse Contractive Autoencoder (SparseCAE) on MNIST. The experiment is structured in four stages:
  - **Data Loading & Preprocessing**: MNIST is loaded from raw binary files and standardized via `StandardScaler` fitted on the training split.
  - **Model Architecture & Training**: A fully-connected autoencoder with a 4-dimensional bottleneck is trained using PyTorch Lightning. The encoder bottleneck applies `ReLU → Tanh` to constrain activations to `[0, 1]`, and training is regularized with both a sparsity penalty and a contractive penalty (see *Model* below).
  - **Reconstruction Visualization**: Interactive widget to inspect original vs. reconstructed images alongside the 4-dimensional latent activation bar chart for any training or test sample.
  - **Latent Space Exploration**: Interactive widget with four sliders (`z[0]`–`z[3]`), each in `[0, 1]`, for manually setting the latent code and observing the decoded output in real time.

### Model

The `SparseCAE` is defined in `SparseCAE.py` and extends `pl.LightningModule`. Its design combines three losses:

- **Reconstruction** (`MSE`): standard pixel-level fidelity.
- **Sparsity** (`λ_sparse · ||z||₁`): L1 penalty on the latent activations. This discourages the model from using all dimensions simultaneously, pushing most "bits" toward zero.
- **Contractive** (`λ_cae · ||∂z/∂x||²_F`): Frobenius norm of the encoder Jacobian, computed via autograd. This penalizes sensitivity of the latent code to input perturbations. When combined with the `ReLU → Tanh` bottleneck, the contractive term pushes each activation toward the saturation regions of `Tanh` (near 0 or 1), where the derivative is small — effectively discouraging intermediate values.

The interaction between the two regularizers is the core inductive bias of this experiment: sparsity determines *how many* bits fire, and contractiveness determines *how cleanly* they saturate.

### Results

![](/assets/AE_MNIST/reconstructed_vs_original.png)

![](/assets/AE_MNIST/heatmap.png)

I noticed that moving z\[1\] affected the witdth of the number, indicating that the model learned the variation in people's hand drawn numbers rather than the numbers themselves.
![](/assets/AE_MNIST/zero_thick.png)![](/assets/AE_MNIST/zero_thin.png)

### Conclusions

This experiment was not driven by performance targets. The motivation was purely exploratory: can an autoencoder be nudged, through architectural choices and regularization, into learning a latent space where each dimension acts as a manual switch? The ReLU → Tanh bottleneck combined with sparsity and contractive regularization was designed to produce near-binary latent codes without explicitly discretizing them during training.

### Data

MNIST is downloaded via `kagglehub`. No other external data is used. All preprocessing (flattening and standardization) is done in-notebook and is not persisted to disk.