# TATA TechPulse – AI & Data Science Assignments

This repository contains 8 practical assignments covering Artificial Intelligence, Machine Learning, Deep Learning, Computer Vision, and Natural Language Processing using Python.

## Assignments

### 1. Car Mileage Estimation
A machine learning regression project that predicts vehicle mileage.

**Technologies:** Python, NumPy, Pandas, Scikit-learn

**Results:**
- MAE: 1.93
- RMSE: 2.40
- R² Score: 0.806

### 2. Simulated Driving Agent
A simulation-based driving agent implemented using evolutionary optimization concepts.

**Concepts:**
- Fitness Evaluation
- Selection
- Crossover
- Mutation
- Evolutionary Optimization

**Result:**
- Initial Fitness: 55.61
- Best Fitness: 74.22

### 3. Data Cleaning & Preprocessing
Demonstrates common data preprocessing and cleaning techniques.

**Operations:**
- Missing Value Handling
- Data Cleaning
- Feature Selection
- Data Transformation
- Preprocessing

**Result:**
- Processed Dataset Shape: `(6, 4)`

### 4. Vehicle Price Prediction
Predicts vehicle prices using multiple regression algorithms.

**Models Used:**
- Linear Regression
- Ridge Regression
- Random Forest Regression

**Results:**

| Model | MAE | R² Score |
|---|---:|---:|
| Linear Regression | 1424.47 | 0.931 |
| Ridge Regression | 1407.34 | 0.932 |
| Random Forest | 1717.00 | 0.897 |

### 5. Predictive Maintenance
A machine learning classification project for predicting potential equipment failures.

**Technologies:**
- Python
- Pandas
- Scikit-learn
- Random Forest

**Evaluation Metrics:**
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC-AUC

**Results:**
- Accuracy: 86%
- ROC-AUC: 0.536

### 6. Traffic Sign Classification using CNN
A Convolutional Neural Network project for traffic sign image classification.

**Technologies:**
- Python
- TensorFlow
- Keras
- CNN
- Image Processing

**Architecture:**

Input Image → Rescaling → Conv2D → MaxPooling → Conv2D → MaxPooling → Conv2D → MaxPooling → Flatten → Dropout → Dense → Output

**Results:**
- Images: 306
- Classes: 2
- Epochs: 5
- Training Accuracy: 97.14%
- Validation Accuracy: 95.08%

### 7. Pedestrian Detection
A computer vision project that detects pedestrians in images.

**Technologies:**
- Python
- OpenCV
- HOG Descriptor
- SVM-based Pedestrian Detector

**Result:**
- Input Image: `input.png`
- Output Image: `detected.png`
- Pedestrians Detected: 6

### 8. Sentiment Analysis using LSTM
A Natural Language Processing project that classifies text as positive or negative using an LSTM neural network.

**Technologies:**
- Python
- PyTorch
- LSTM
- Natural Language Processing
- Tokenization
- Word Embeddings

**Results:**
- Accuracy: 70%
- Precision: 66.67%
- Recall: 80%
- F1 Score: 72.73%

**Example Predictions:**

- "excellent quality and great service" → Positive
- "terrible product and poor service" → Negative
- "this was a horrible experience" → Negative

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- TensorFlow
- Keras
- PyTorch
- OpenCV
- Matplotlib
- Seaborn
- Git
- GitHub

## Project Structure

```text
TATA-TechPulse/
│
├── Assignment 1/
├── Assignment 2/
├── Assignment 3/
├── Assignment 4/
├── Assignment 5/
├── Assignment 6/
├── Assignment 7/
├── Assignment 8/
│
├── requirements.txt
├── .gitignore
└── README.md
