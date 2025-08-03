MNIST Digit Classifier
A simple neural network for handwritten digit recognition using TensorFlow/Keras




Overview
This project trains a feedforward neural network to classify handwritten digits (0–9) from the MNIST dataset.

It’s a beginner-friendly machine learning project designed to demonstrate:

Loading and preprocessing image data

Building a neural network with Keras

Training, evaluating, and visualizing model performance

Model accuracy: ~97–98% on the test set after 5 epochs.

Dataset
Name: MNIST (Modified National Institute of Standards and Technology database)

Size: 70,000 grayscale images (60,000 training, 10,000 testing)

Image dimensions: 28×28 pixels

Classes: 10 (digits 0–9)

Model Architecture
Flatten Layer – Converts 28×28 images into a 784-element vector

Dense Hidden Layer – 128 neurons with ReLU activation

Dropout Layer – 20% dropout rate to reduce overfitting

Dense Output Layer – 10 neurons with Softmax activation for multi-class probabilities

Optimizer: Adam
Loss Function: Sparse categorical cross-entropy

Installation & Setup
1. Clone the Repository
bash
Copy code
git clone https://github.com/Jesse-Oh-3709/mnist_project.git
cd mnist_project
2. Install Dependencies
bash
Copy code
pip install tensorflow matplotlib numpy
3. Run the Notebook
bash
Copy code
jupyter notebook MNIST_Digits.ipynb
Usage
Open the notebook.

Run all cells to:

Load and preprocess the dataset.

Train the model for 5 epochs.

Evaluate accuracy on the test set.

Visualize predictions with Matplotlib.

Sample Output
Example prediction:


The model predicts "4" with 98% confidence.

Next Steps & Improvements
Upgrade to a Convolutional Neural Network (CNN): Better performance for image tasks.

Hyperparameter tuning: Experiment with different optimizers, learning rates, and hidden layer sizes.

Data augmentation: Slight rotations and shifts to make the model more robust.

Model deployment: Export using model.save() and integrate into a web or mobile app.

Why This Project?
MNIST is the “Hello World” of deep learning. Completing this project builds a foundation for:

Computer vision applications

More advanced networks (CNNs, RNNs)

Real-world deployment

Author: Jesse Oh
If you like this project, feel free to ⭐ the repository!
