# 🤖 MNIST Handwritten Digit Recognition

A machine learning learning project featuring both a Jupyter notebook implementation and an interactive web application for handwritten digit recognition using neural networks.

## 🎯 Project Overview & Learning Journey

This project represents my introduction to machine learning and neural networks using the famous MNIST dataset. It includes:

- **📓 Jupyter Notebook**: My original implementation with data analysis, model training, and evaluation
- **🌐 Interactive Web App**: AI-assisted web application for real-time digit recognition
- **🐳 Docker Support**: Deployment configuration 
- **📊 Model Performance**: 97%+ accuracy on test data

### 🤖 AI Assistance Disclosure

**Transparency Note**: While the core MNIST neural network implementation in the Jupyter notebook (`MNIST_Digits.ipynb`) represents my original learning work, the web application (`app.py`, HTML interface, Docker configuration, and this README) were created with significant assistance from Claude AI. This project serves as both a learning exercise in machine learning fundamentals and an exploration of AI-assisted development.

## 🚀 Features

### Jupyter Notebook (`MNIST_Digits.ipynb`) - Original Work
- ✅ My hands-on exploration of the MNIST dataset
- ✅ Learning neural network architecture design
- ✅ Understanding model training and evaluation
- ✅ Practice with TensorFlow/Keras fundamentals
- ✅ Data visualization and analysis techniques

### Interactive Web Application - AI-Assisted Development
- 🎨 **Drawing Canvas**: Real-time digit drawing interface
- 🤖 **AI Recognition**: Instant predictions with confidence scores
- 📊 **Probability Visualization**: Interactive charts and analytics
- 📱 **Responsive Design**: Cross-platform compatibility
- 🔄 **Model Management**: Training and architecture viewing
- ⚡ **Production Features**: Docker deployment and optimization

*Note: The web application was developed with AI assistance to explore modern ML deployment practices and full-stack development workflows.*

## 🏗️ Architecture

### Neural Network Model
```
Input Layer (784 neurons) → Flatten 28x28 images
Hidden Layer (128 neurons) → ReLU activation  
Dropout Layer (20%) → Prevent overfitting
Output Layer (10 neurons) → Softmax for digit classification
```

### Technology Stack
- **Backend**: Flask (Python web framework)
- **ML Framework**: TensorFlow/Keras
- **Frontend**: HTML5, CSS3, JavaScript
- **Image Processing**: OpenCV, PIL
- **Deployment**: Docker, Gunicorn

## 📊 Performance Metrics

- **Model Accuracy**: 97.23% on test set
- **Training Time**: ~2-3 minutes (5 epochs)
- **Prediction Speed**: ~50ms per digit
- **Model Size**: ~500KB saved file
- **Dataset**: 70,000 images (60K train, 10K test)

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

3. **Run all cells** to see the complete analysis

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
├── 📓 MNIST_Digits.ipynb        # Jupyter notebook with complete analysis
├── 🌐 app.py                   # Flask web application
├── 📁 templates/
│   └── index.html              # Web interface
├── 🐳 Dockerfile               # Container configuration
├── 📋 requirements.txt         # Python dependencies
├── 📖 README.md                # This file
├── 🤖 mnist_model.h5           # Trained model (created after first run)
└── 📁 .git/                    # Git version control
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+ (3.9-3.11 recommended)
- pip or conda package manager
- Modern web browser

### Dependencies
```bash
Flask==2.3.3           # Web framework
tensorflow==2.13.0     # Machine learning
numpy==1.24.3          # Numerical computing
opencv-python==4.8.0   # Image processing
Pillow==10.0.0         # Image handling
gunicorn==21.2.0       # Production server
```

### First-Time Setup
```bash
# Clone repository
git clone https://github.com/Jesse-Oh-3709/mnist_project.git
cd mnist_project

# Install dependencies
pip install -r requirements.txt

# Run web application
python app.py
```

## 🎮 How to Use the Web App

### Drawing Interface
1. **Draw a digit** (0-9) on the white canvas using your mouse or finger
2. **Click "Predict"** to get AI recognition
3. **View results**: See the predicted digit and confidence score
4. **Check probabilities**: View the probability distribution for all digits
5. **Clear canvas** to try again

### Additional Features
- **🔄 Retrain Model**: Click to retrain with fresh MNIST data
- **ℹ️ Model Info**: View neural network architecture and parameters
- **📱 Mobile Support**: Works on smartphones and tablets

## 📈 Model Training Details

### Dataset Information
- **Source**: MNIST Database of Handwritten Digits
- **Training Images**: 60,000 (28x28 grayscale)
- **Test Images**: 10,000 (28x28 grayscale)
- **Classes**: 10 digits (0-9)
- **Preprocessing**: Normalized to [0,1] range

### Training Configuration
- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy
- **Batch Size**: 128
- **Epochs**: 5
- **Validation Split**: 10,000 test images

### Expected Results
```
Epoch 1/5: 79.57% → 94.27% accuracy
Epoch 2/5: 93.90% → 95.81% accuracy
Epoch 3/5: 95.23% → 96.61% accuracy
Epoch 4/5: 96.50% → 97.02% accuracy
Epoch 5/5: 97.12% → 97.23% accuracy

Final Test Accuracy: 97.23%
```

## 🚀 Deployment Options

### Local Development
```bash
python app.py
# Access at http://localhost:5001
```

### Docker Deployment
```bash
docker build -t mnist-webapp .
docker run -p 5001:5001 mnist-webapp
```

### Cloud Deployment

#### Heroku
```bash
# Add Procfile
echo "web: gunicorn app:app" > Procfile

# Deploy
heroku create your-mnist-app
git push heroku main
```

#### Railway/Render
- Connect your GitHub repository
- Automatic deployment from `main` branch
- Uses `Dockerfile` for containerized deployment

#### Google Cloud Run
```bash
# Build and deploy
gcloud builds submit --tag gcr.io/PROJECT-ID/mnist-webapp
gcloud run deploy --image gcr.io/PROJECT-ID/mnist-webapp --platform managed
```

## 🔧 Troubleshooting

### Common Issues

#### TensorFlow Installation
```bash
# For Mac M1/M2
pip install tensorflow-macos

# For general compatibility issues
pip install tensorflow==2.13.0
```

#### Port Already in Use
```bash
# Check what's using port 5001
lsof -i :5001

# Or edit app.py to use different port
# Change: app.run(debug=True, host='0.0.0.0', port=5002)
```

#### Missing Model File
- On first run, the app will automatically download MNIST data and train the model
- This takes 2-3 minutes but only happens once
- Subsequent runs load the saved `mnist_model.h5` instantly

#### Canvas Not Working
- Ensure you're using a modern browser with HTML5 Canvas support
- On mobile, try refreshing the page if touch events don't work initially

## 📚 Learning Outcomes & Skills Developed

### Machine Learning Fundamentals (Original Work)
- **Neural Network Architecture**: Understanding layers, neurons, and activation functions
- **Data Preprocessing**: Normalizing and preparing image data for training
- **Model Training**: Learning epochs, batch sizes, and optimization techniques
- **Performance Evaluation**: Accuracy metrics, loss functions, and validation
- **TensorFlow/Keras**: Hands-on experience with ML frameworks

### Full-Stack Development (AI-Assisted Learning)
- **Web Application Architecture**: Flask backend with HTML/JavaScript frontend
- **Real-time Image Processing**: Canvas drawing to model prediction pipeline
- **Deployment Practices**: Docker containerization and production considerations
- **User Interface Design**: Creating intuitive ML application interfaces
- **Version Control**: Git workflow and project documentation

### Development Philosophy
This project represents a hybrid learning approach - combining traditional hands-on ML education with modern AI-assisted development to understand both the fundamentals and current industry practices.

## 🤝 Contributing & Learning

This project is open for contributions and serves as a learning resource for others beginning their ML journey.

### What I Learned
- **ML Fundamentals**: Neural networks, training, and evaluation
- **Modern Development**: AI-assisted coding and deployment practices
- **Problem Solving**: Debugging, troubleshooting, and iteration

### For Other Learners
- The Jupyter notebook provides a step-by-step introduction to MNIST classification
- The web application demonstrates how ML models can be deployed in real applications
- The combination shows both traditional learning and modern development workflows

## 📚 Additional Learning Resources

### Understanding the Core Concepts
- **Neural Networks**: [Deep Learning Basics](https://www.deeplearningbook.org/)
- **TensorFlow/Keras**: [Official Tutorials](https://www.tensorflow.org/tutorials)
- **MNIST Dataset**: [Original Paper](http://yann.lecun.com/exdb/mnist/)

### Next Steps for Learning
- Try different neural network architectures (CNN, RNN)
- Experiment with other datasets (Fashion-MNIST, CIFAR-10)
- Learn about data augmentation and regularization techniques
- Explore more advanced ML concepts and frameworks

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- **My Learning Journey**: This project represents my first steps into machine learning
- **MNIST Dataset**: Yann LeCun, Corinna Cortes, Christopher J.C. Burges - for creating this foundational dataset
- **Claude AI**: For assistance in developing the web application and exploring modern ML deployment practices
- **TensorFlow Team**: For the excellent ML framework that made learning accessible
- **Open Source Community**: For tutorials, documentation, and inspiration
- **ML Education Community**: For making machine learning approachable for beginners

## 📬 Contact

- **GitHub**: [@Jesse-Oh-3709](https://github.com/Jesse-Oh-3709)
- **Project Link**: [https://github.com/Jesse-Oh-3709/mnist_project](https://github.com/Jesse-Oh-3709/mnist_project)

---

### 🎯 Quick Commands Reference

```bash
# Setup
git clone https://github.com/Jesse-Oh-3709/mnist_project.git
cd mnist_project
pip install -r requirements.txt

# Run
python app.py                    # Start web app
jupyter notebook                 # Open notebook

# Deploy
docker build -t mnist-webapp .   # Build container
docker run -p 5001:5001 mnist-webapp  # Run container

# Development
git add .                        # Stage changes
git commit -m "Your message"     # Commit changes
git push origin main             # Push to GitHub
```

**Ready to recognize some digits? Start drawing! 🎨🤖**
