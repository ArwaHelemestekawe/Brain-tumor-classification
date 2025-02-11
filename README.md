# Brain Tumor Classification using Machine Learning

## Overview
This project implements a machine learning solution for multi-class brain tumor classification using scikit-learn. The model classifies brain tumor MRI images into four different classes, achieving an accuracy of 85% using Support Vector Classification (SVC).

## Dataset
- Source: Kaggle Brain Tumor MRI Dataset
- Total Images: 3,264
  - Training set: 2,870 images
  - Test set: 394 images
- Classes: 4 different types of brain tumors
- Image format: Grayscale, resized to 150x150 pixels

## Prerequisites
```
numpy
pandas
scikit-learn
opencv-python
matplotlib
```

## Methodology

### Data Preprocessing
1. Image loading and resizing to 150x150 pixels
2. Conversion to grayscale
3. Feature standardization using MinMaxScaler
4. Train-test split

### Model Development
- Implemented and compared multiple classification models
- Best performing models:
  1. Support Vector Classification (SVC)
  2. Random Forest Classifier

### Hyperparameter Tuning
Performed GridSearchCV for SVC with the following parameters:
- C: [1, 10, 20]
- Kernel: 'rbf'
- Gamma: [1, 0.1, 0.01]

## Results
- Final Model: Support Vector Classification (SVC)
- Accuracy: 85%
- Model evaluation metrics include:
  - F1-score
  - Precision
  - Recall
  - Confusion Matrix

## Model Selection
The project initially experimented with multiple classification models. SVC and Random Forest were selected as the final candidates due to their superior performance. The SVC model with optimized hyperparameters was chosen as the final model based on its highest accuracy.

## Future Improvements
- Experiment with deep learning approaches
- Collect more training data
- Implement cross-validation
- Try different preprocessing techniques


