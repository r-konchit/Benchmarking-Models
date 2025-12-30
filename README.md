# Benchmarking Linear Classification Models

## Overview
This repository benchmarks linear classification models implemented from scratch
and evaluates their performance across multiple text classification tasks.

The goal is to understand how classic linear models behave under different datasets,
feature representations, and training without using prebuilt ML libraries.

## Models Implemented
- Perceptron
- Logistic Regression
- Linear classifiers using bag-of-letters features

## Tasks & Datasets
- **Language Identification** (English vs Dutch)
- **Spam Detection**
- **Universal Declaration of Human Rights** text classification

Each task shows how feature engineering and optimization choices impact
classification performance.

## How to Run
Each task directory contains a Jupyter notebook demonstrating:
- Data preprocessing and feature extraction
- Model training
- Evaluation and comparison

Open a notebook and run all cells to reproduce results.

## countLetters(words):
This returns an array of length 26 which is counting the occurrences of each letter from a to z in the input string.  
Non alphabetic characters are ignored, and the uppercase is converted to lowercase

## make_X_and_y(text, language)
Converts text into a matrix of letter frequency vectors 'X' and a label vector 'y.  
Labels are 1 for English and -1 for Dutch, based on the inputted language string

## unison_shuffled_copies(a, b):
Essentially just shuffles two np arrays

## perceptronTrain(X, y, num_epochs):
This implements the Perceptron training algorithm. It iterates over the training data for a 
specified number of epochs to learn the weights and bias.

## countLetters(words):
This returns an np array length of 26, where each index refers to the letter count.

## make_X_and_y(text, language):
Makes an X for both english and dutch


## classify(weights, bias, example):
This does the dot product of input features and weights, adds bias, and returns a class prediction.
Returns 1 if the activation is positive or-1 if negative, and 0 if zero.


## runTests(weights, bias, X_test, y_test):
Just loops through all test samples, then classifies them, and and then counts how many predictions 
are correct. Returns the accuracy as number of correct predictions divided by the total test samples

### Functions used from the Scikit learn:

### StandardScaler()
Creates a StandardScaler object which will scale features to have zero mean and unit variance.

### scaler.fit_transform(X)
Fits the scaler on the training data X which finds the mean and std
Returns the scaled version of X.

### scaler.transform(X_test)
Basically applies the same scaling from the training data to the test set X_test

### LogisticRegression(penalty="l2", C=1.0, solver="saga", max_iter=5000)
-  penalty="l2": L2 regularization

 - C=1.0: Regularization factor

 - solver="saga": on tutorial

 - max_iter=5000: lets up to 5000 iterations for the solver to converge

### classifier.fit(X_train_fitted, y)
Trains the logistic regression model using the scaled training data from X_train_fitted and labels the y.

### train_test_split(X, y, test_size=0.1, random_state=42)

X: input data
y: Label which is the target 

test_size=0.1: 90% of the data goes into the test set, and the other 10 into training

random_state=42: Sets the random seed for so its the same split each time

### precision_recall_fscore_support(y_test, y_pred, labels)
returns
precision[]
recall[]
f1Score[]
