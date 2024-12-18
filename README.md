# Multilayer Perceptron Notebook

This repository contains a Jupyter Notebook that implements and evaluates machine learning models, focusing on deep learning architectures like Multilayer Perceptrons (MLPs). The notebook demonstrates data loading, preprocessing, model creation, training, and evaluation using frameworks like PyTorch and libraries such as NumPy and Matplotlib.

## Features

- **Data Handling**: 
  - Loads data using libraries like `numpy` and `pandas`.
  - Includes examples with datasets like MNIST (loaded via `fetch_openml`).

- **Model Definitions**:
  - Implements an MLP class from scratch.
  - Uses PyTorch for defining and training neural networks (`MLP_pytorch`).
  
- **Training and Evaluation**:
  - Configurable activation functions and network sizes.
  - Includes helper functions to manage randomness and evaluate model performance.
  - Tracks training metrics including **loss** and **accuracy** at each epoch.
  - Evaluates model performance on both training and testing datasets.

- **Visualization**:
  - Generates plots to visualize training results, including:
    - Loss vs. Epoch curves
    - Accuracy vs. Epoch curves
  - Supports visualizing approximation functions (e.g., `relu_approximation`).

## Requirements

Ensure the following libraries are installed:

- `numpy`
- `pandas`
- `matplotlib`
- `torch`
- `scikit-learn`
- `gdown` (if applicable for external data loading)

Install all dependencies using the following command:

```bash
pip install numpy pandas matplotlib torch scikit-learn gdown
```

## Usage

1. Clone the repository and navigate to the project directory:

```bash
git clone <repository_url>
cd <repository_directory>
```

2. Launch the Jupyter Notebook:

```bash
jupyter notebook multilayer_perceptron.ipynb
```

3. Follow the instructions provided in the notebook cells to:
   - Load the data.
   - Initialize and train models.
   - Visualize and interpret results.

## Example Outputs

- **Data Visualization**: Displays samples from datasets like MNIST.
- **Training Progress**: 
  - **Loss**: Observe how the loss decreases across epochs, indicating the model is learning.
  - **Accuracy**: Monitor accuracy improvements on training and validation datasets.
- **Activation Approximations**: Visualizes activation functions and their approximations (e.g., ReLU variants).

### Results Summary

- **Accuracy**: Achieved a final accuracy of approximately 89% on the test dataset 
- **Loss**: Final training loss was approximately 3.8 .
- **Model Configuration**:
  -  Input layer (784 nodes) → Hidden layers (128, 64 nodes) → Output layer (10 nodes).
  - Activation Functions: Sigmoid, ReLU,Elu, Tanh and others were evaluated.

