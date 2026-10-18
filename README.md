# Credit Risk Prediction Model

## Project Overview

This project predicts the risk of loan default using machine learning techniques.

The project uses financial and personal information such as income, loan amount, loan interest rate, employment length, loan grade, and previous credit history.

## Machine Learning Models

The following models were implemented:

- Logistic Regression
- Decision Tree
- Random Forest

## Data Preprocessing

- Removed duplicate records
- Handled missing values
- Converted categorical features using One-Hot Encoding
- Split the dataset into 80% training and 20% testing data

## Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

## Best Model

Random Forest performed best among the tested models.

### Random Forest Results

- Accuracy: 93%
- Precision: 97%
- Recall: 71%
- F1 Score: 82%
- ROC-AUC: 93%

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib

## Project Outcome

The trained Random Forest model can be used to predict loan default risk based on the provided financial and loan-related information.