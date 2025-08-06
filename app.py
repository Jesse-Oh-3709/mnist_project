from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow import keras
import matplotlib.pyplot as plt
import cv2
import base64
from io import BytesIO
from PIL import Image
import os
import json

app = Flask(__name__)

# Global variable to store the model
model = None

def create_model():
    """Create the MNIST neural network model"""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])

    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def train_model():
    """Train the model on MNIST data"""
    global model
    
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Reshape to add channel dimension (28, 28) → (28, 28, 1)
    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0


    
    # Create model
    model = create_model()
    
    # Train model
    print("Training model...")
    history = model.fit(x_train, y_train, epochs=5, 
                    validation_split=0.1, batch_size=64)

    
    # Save model
    model.save('mnist_model.h5')
    
    # Evaluate model
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"\nTest accuracy: {test_acc:.4f}")

    plt.plot(history.history['accuracy'], label='train acc')
    plt.plot(history.history['val_accuracy'], label='val acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('CNN Training Accuracy')
    plt.show()


    
    return history

def load_model():
    """Load the trained model"""
    global model
    
    if os.path.exists('mnist_model.h5'):
        print("Loading existing model...")
        model = keras.models.load_model('mnist_model.h5')
    else:
        print("No existing model found. Training new model...")
        train_model()

def preprocess_image(image_data):
    """Preprocess the drawn image for prediction"""
    # Decode base64 image
    image_data = image_data.split(',')[1]  # Remove data:image/png;base64, prefix
    image = Image.open(BytesIO(base64.b64decode(image_data)))
    
    # Convert to grayscale
    image = image.convert('L')
    
    # Convert to numpy array
    image_array = np.array(image)
    
    # Resize to 28x28
    image_array = cv2.resize(image_array, (28, 28))
    
    # Invert colors (make background black, digit white)
    image_array = 255 - image_array
    
    # Normalize
    image_array = image_array.astype('float32') / 255.0
    
    # Reshape for model input (CNN expects 4D: batch_size, height, width, channels)
    image_array = image_array.reshape(1, 28, 28, 1)
    
    return image_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image data from request
        data = request.get_json()
        image_data = data['image']
        
        # Preprocess image
        processed_image = preprocess_image(image_data)
        
        # Make prediction
        predictions = model.predict(processed_image)
        predicted_digit = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        # Get all probabilities
        probabilities = [float(p) for p in predictions[0]]
        
        return jsonify({
            'predicted_digit': int(predicted_digit),
            'confidence': confidence,
            'probabilities': probabilities
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/retrain', methods=['POST'])
def retrain():
    """Retrain the model"""
    try:
        history = train_model()
        return jsonify({'message': 'Model retrained successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model_info')
def model_info():
    """Get model information"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    # Get model summary as string
    summary_list = []
    model.summary(print_fn=lambda x: summary_list.append(x))
    summary_string = '\n'.join(summary_list)
    
    return jsonify({
        'summary': summary_string,
        'total_params': model.count_params()
    })

if __name__ == '__main__':
    # Load or train model on startup
    load_model()
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5001)
