# **MNIST Digit Classifier**  
*A simple neural network for handwritten digit recognition using TensorFlow/Keras*  

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)  
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)  
![Status](https://img.shields.io/badge/Status-Completed-success)  

---

## **Overview**  
This project trains a **feedforward neural network** to classify handwritten digits (0–9) from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/).  

It’s a beginner-friendly machine learning project designed to demonstrate:  
- Loading and preprocessing image data  
- Building a neural network with Keras  
- Training, evaluating, and visualizing model performance  

**Model accuracy:** ~97–98% on the test set after 5 epochs.

---

## **Dataset**  
- **Name:** MNIST (Modified National Institute of Standards and Technology database)  
- **Size:** 70,000 grayscale images (60,000 training, 10,000 testing)  
- **Image dimensions:** 28×28 pixels  
- **Classes:** 10 (digits 0–9)  

---

## **Model Architecture**  
- **Flatten Layer** – Converts 28×28 images into a 784-element vector  
- **Dense Hidden Layer** – 128 neurons with **ReLU** activation  
- **Dropout Layer** – 20% dropout rate to reduce overfitting  
- **Dense Output Layer** – 10 neurons with **Softmax** activation for multi-class probabilities  

**Optimizer:** Adam  
**Loss Function:** Sparse categorical cross-entropy  

---

## **Installation & Setup**  

### **1. Clone the Repository**
```bash
git clone https://github.com/Jesse-Oh-3709/mnist_project.git
cd mnist_project
