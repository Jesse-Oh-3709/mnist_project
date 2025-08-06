# 🤖 MNIST Handwritten Digit Recognition

A machine learning project featuring both a Jupyter notebook implementation and an interactive web application for handwritten digit recognition using **Convolutional Neural Networks (CNNs)**.

## 🎯 Project Overview & Learning Journey

This project represents my introduction to machine learning and CNNs using the famous MNIST dataset. It includes:

* **📓 Jupyter Notebook**: My original CNN-based implementation with data analysis, model training, and evaluation
* **🌐 Interactive Web App**: AI-assisted web application for real-time digit recognition
* **🐳 Docker Support**: Deployment configuration
* **📊 Model Performance**: \~99% accuracy on test data

### 🤖 AI Assistance Disclosure

**Transparency Note**: While the core CNN implementation in the Jupyter notebook (`MNIST_Digits.ipynb`) represents my original learning work, the web application (`app.py`, HTML interface, Docker configuration, and this README) were created with significant assistance from Claude AI. This project serves as both a learning exercise in deep learning fundamentals and an exploration of AI-assisted development.

## 🚀 Features

### Jupyter Notebook (`MNIST_Digits.ipynb`) - Original Work

* ✅ Hands-on exploration of the MNIST dataset
* ✅ Learning and implementing CNN architecture
* ✅ Understanding model training and evaluation
* ✅ Practice with TensorFlow/Keras fundamentals
* ✅ Data visualization and analysis techniques

### Interactive Web Application - AI-Assisted Development

* 🎨 **Drawing Canvas**: Real-time digit drawing interface
* 🤖 **AI Recognition**: Instant predictions with confidence scores
* 📊 **Probability Visualization**: Interactive charts and analytics
* 📱 **Responsive Design**: Cross-platform compatibility
* 🔄 **Model Management**: Training and architecture viewing
* ⚡ **Production Features**: Docker deployment and optimization

*Note: The web application was developed with AI assistance to explore modern ML deployment practices and full-stack development workflows.*

## 🏗️ Architecture

### CNN Model Architecture

```
Input Layer (28x28x1 grayscale images)
→ Conv2D (32 filters, 3x3 kernel, ReLU)
→ MaxPooling2D (2x2 pool size)
→ Conv2D (64 filters, 3x3 kernel, ReLU)
→ MaxPooling2D (2x2 pool size)
→ Flatten
→ Dense (64 neurons, ReLU)
→ Dropout (0.5)
→ Dense (10 neurons, Softmax)
```

### Technology Stack

* **Backend**: Flask (Python web framework)
* **ML Framework**: TensorFlow/Keras
* **Frontend**: HTML5, CSS3, JavaScript
* **Image Processing**: OpenCV, PIL
* **Deployment**: Docker, Gunicorn

## 📊 Performance Metrics

* **Model Accuracy**: \~99% on test set
* **Training Time**: \~2-3 minutes (5 epochs)
* **Prediction Speed**: \~50ms per digit
* **Model Size**: \~500KB saved file
* **Dataset**: 70,000 images (60K train, 10K test)

## 🚀 Quick Start

### Option 1: Web Application

1. **Clone the repository**:

   ```bash
   git clone https://github.com/Jesse-Oh-3709/mnist_project.git
   cd mnist_project
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the web app**:

   ```bash
   python app.py
   ```

4. **Open your browser** and navigate to `http://localhost:5001`

5. **Draw and predict** digits in real-time! 🎨

### Option 2: Jupyter Notebook

1. **Start Jupyter**:

   ```bash
   jupyter notebook
   ```

2. **Open** `MNIST_Digits.ipynb`

3. **Run all cells** to see the complete CNN implementation and analysis

### Option 3: Docker

1. **Build the image**:

   ```bash
   docker build -t mnist-webapp .
   ```

2. **Run the container**:

   ```bash
   docker run -p 5001:5001 mnist-webapp
   ```

3. **Access** at `http://localhost:5001`

## 📁 Project Structure

```
mnist_project/
📝 MNIST_Digits.ipynb        # Jupyter notebook with CNN implementation
🌐 app.py                   # Flask web application
📁 templates/
    └ index.html              # Web interface
📃 Dockerfile               # Container configuration
📋 requirements.txt         # Python dependencies
📖 README.md                # This file
🤖 mnist_model.h5           # Trained CNN model
📁 .git/                    # Git version control
```
