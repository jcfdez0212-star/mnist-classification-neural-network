# Deep Neural Network for MNIST Classification

## Overview

This project implements a deep neural network using TensorFlow to classify handwritten digits from the MNIST dataset. The MNIST dataset consists of 70,000 grayscale images (28x28 pixels each) of handwritten digits from 0 to 9, making this a 10-class classification problem.

## Project Goal

Build and train a neural network with 2 hidden layers that can accurately recognize which digit (0-9) is written in a given image.

## Dataset

- **Source:** MNIST dataset, loaded via `tensorflow_datasets`
- **Total samples:** 70,000 images (60,000 for training/validation, 10,000 for testing)
- **Image size:** 28x28 pixels, single channel (grayscale)
- **Split used:**
  - 90% of the training set for training
  - 10% of the training set for validation
  - Full test set for final evaluation

## Preprocessing

1. **Scaling:** Pixel values are cast to `float32` and divided by 255 so all inputs fall between 0 and 1.
2. **Shuffling:** Training data is shuffled using a buffer size of 10,000 to avoid any ordering bias.
3. **Batching:**
   - Training data is batched with a batch size of 100.
   - Validation and test data are each batched as a single batch (using the full validation/test set size), since batching isn't needed for forward passes without backpropagation.

## Model Architecture

```
Input Layer    → Flatten (28x28x1 → 784)
Hidden Layer 1 → Dense, 50 units, ReLU activation
Hidden Layer 2 → Dense, 50 units, ReLU activation
Output Layer   → Dense, 10 units, Softmax activation
```

- **Input size:** 784 (flattened 28x28 image)
- **Hidden layer size:** 50 neurons per layer
- **Output size:** 10 (one neuron per digit class)

## Training Configuration

- **Optimizer:** Adam
- **Loss function:** Sparse categorical crossentropy
- **Metric:** Accuracy
- **Epochs:** 5
- **Batch size:** 100

## Results

The trained model achieves a test accuracy of approximately **97.5%**, which is a solid result for a simple feedforward neural network with only 2 hidden layers of 50 neurons each.

## Requirements

- Python
- `numpy`
- `tensorflow`
- `tensorflow_datasets`

## How to Run

1. Install the required packages:
   ```bash
   pip install numpy tensorflow tensorflow-datasets
   ```
2. Open and run `TF_MNIST.ipynb` cell by cell in Jupyter Notebook or JupyterLab.
3. The notebook will automatically download the MNIST dataset on first run (via `tensorflow_datasets`), preprocess it, build the model, train it, and evaluate it on the test set.

## Possible Improvements

- Experiment with a larger hidden layer size or additional hidden layers.
- Try different activation functions.
- Tune the number of epochs, batch size, or learning rate.
- Add dropout or regularization to reduce overfitting.
- Use convolutional layers (CNN) instead of a fully connected network for potentially higher accuracy.
