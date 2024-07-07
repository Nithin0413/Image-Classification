# Image Classification with TensorFlow #
This project involves image classification using TensorFlow on the CIFAR-10 dataset, demonstrating the implementation of Artificial Neural Networks (ANN) and Convolutional Neural Networks (CNN).

## Introduction ##
The project showcases the classification of images into 10 categories using ANN and CNN models. The categories include airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

## Data Preprocessing ##
Images from the CIFAR-10 dataset are normalized to values between 0 and 1.

## Classification Using Artificial Neural Network (ANN)##
The ANN model includes:
  * Flatten layer for image reshaping
  
  * Dense layers with ReLU activation
  
  * Output layer with softmax activation
  
  * Optimizer: Stochastic Gradient Descent (SGD)
  
  * Loss function: Sparse categorical cross-entropy
  
  * Metrics: Accuracy

## Classification Using Convolutional Neural Network (CNN) ##
The CNN model includes:
  * Conv2D layers with ReLU activation.
    
  * MaxPooling layers for spatial reduction.
    
  * Flatten layer for data preparation.
    
  * Dense layers with ReLU and softmax activations.
    
  * Optimizer: Adam optimizer.
    
  * Loss function: Sparse categorical cross-entropy.
    
  * Metrics: Accuracy.
    
## Performance Note ##
CNN models generally outperform ANN models in image classification tasks due to their ability to capture spatial features.
