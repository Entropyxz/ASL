# CS-4337
Computer Vision Final Project
# ASL Gesture Recognition with CNN and Mediapipe

This repository contains a project for real-time American Sign Language (ASL) gesture recognition using a Convolutional Neural Network (CNN) and Mediapipe Hands for hand detection and preprocessing.

## Project Overview

The goal of this project is to classify ASL gestures (letters A-Z, and special gestures: `nothing`, `space`, `del`) from live webcam input. The project uses:
- **Mediapipe Hands** for detecting hand landmarks and computing bounding boxes.
- **TensorFlow/Keras CNN model** for gesture classification.

The repository includes:
- Code for preprocessing, gesture detection, and real-time predictions.
- A saved TensorFlow/Keras model (`asl_classification_model.keras`).

## Features

- **Real-Time Webcam Predictions**: Detect and classify ASL gestures in real-time from your webcam.
- **Bounding Box Visualization**: Visual feedback with bounding boxes and confidence scores for predictions.


## Setup Instructions

### Prerequisites

Ensure the following are installed:
- Python (>=3.7)
- TensorFlow
- OpenCV
- Mediapipe
