AI-Based Handwritten Digit Recognition System

A full-stack web application that recognizes handwritten digits (0-9) from an uploaded image.

This project uses a Convolutional Neural Network (CNN) trained on the MNIST dataset. The model is served via a Flask API and consumed by a modern, responsive React.js frontend.

Key Features :

Deep Learning Model: A robust Convolutional Neural Network (CNN) built with TensorFlow/Keras.

Data Augmentation: The model is trained on an augmented MNIST dataset (rotated, shifted, zoomed) to improve real-world accuracy.

Flask Backend API: A simple Python server that loads the trained model and provides a /predict endpoint.

Advanced Preprocessing: The backend intelligently handles user uploads by:

Detecting image type (white-on-black vs. black-on-white).

Applying auto-contrast and thresholding to clean up photos.

Centering and padding the digit to match the MNIST format.

React Frontend: A modern UI with a Home page and a Recognizer tool.


Real-time Predictions: Users can upload an image and instantly receive a prediction and confidence score.

Tech Stack: 

AI / Machine Learning: TensorFlow, Keras, NumPy

Image Processing: Pillow, Scipy

Backend: Flask, Flask-CORS

Frontend: React.js, Axios (for API calls), React-Router-DOM

How to Run This Project : 

1. Set Up and Run the Backend

# Navigate to the root folder
cd AI-Digit-Recognizer

# Create and activate a virtual environment
 Terminal 1
 
python -m venv venv
venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Train the AI model 
# This will create the 'mnist_cnn_model.keras' file
python train_model.py

# Run the Flask server
python app.py
backend will now be running at http://127.0.0.1:5000.



3. Set Up and Run the Frontend
Terminal 2 (New Terminal):

Bash

# Navigate to the frontend folder
cd frontend

# Install Node.js dependencies
npm install

# Run the React development server
npm start
browser will automatically open to http://localhost:3000, where we can use the application.
