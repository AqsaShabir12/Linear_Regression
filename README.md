# Linear Regression from Scratch 

This repository contains a clean implementation of **Linear Regression** using only **NumPy**. The model is trained and tested on a synthetically generated dataset using `sklearn.datasets.make_regression`. The project visualizes the dataset and the prediction line, and evaluates performance using **Mean Squared Error (MSE)** and **variance**.

---

## File Description

- **linear_regression.py**  
  A single Python file that:
  - Implements the `LinearRegression` class using gradient descent  
  - Generates a noisy linear dataset with one feature  
  - Splits the data into training and testing sets  
  - Trains the model  
  - Predicts and evaluates results  
  - Visualizes both raw data and prediction line  

---

## Requirements

Install the required packages with:

```bash
pip install numpy scikit-learn matplotlib
```

## How to Run
Simply execute the script in your terminal:
```bash
python train.py
```
This will:
- Generate a 1D linear dataset with noise
- Split into training and testing data
- Train a linear regression model using gradient descent
- Compute and print Mean Squared Error (MSE), variance, and standard deviation
- Plot:
    - Input data
    - Fitted regression line
    - Train vs test data split

## Dataset

The dataset is generated using the `make_regression` function from `scikit-learn`. It creates a single-feature linear dataset with controlled noise for regression testing and experimentation.
