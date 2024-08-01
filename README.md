# Spam Detection Using Logistic Regression

This project focuses on building a machine learning model to classify emails as spam or ham (not spam) using logistic regression. The model is trained on a dataset of emails and uses text feature extraction techniques to predict the class of new emails.

## Table of Contents
- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Model Evaluation](#model-evaluation)

## Project Overview

The objective of this project is to develop a spam detection system using logistic regression. The system reads email data, processes it, and classifies emails into spam or ham categories. This involves steps such as data collection, pre-processing, feature extraction, model training, and evaluation.

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn

## Dataset
- The dataset used in this project is a CSV file containing emails and their respective labels (spam or ham). The file should be named mail_data.csv and placed in the project directory.

## Project Structure
```sh
spam-detection/
│
├── mail_data.csv          # Dataset file
├── spam_detection.py      # Main script
├── README.md              # Project README
└── requirements.txt       # Python dependencies
```
## Usage
*Import Dependencies*
```sh
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```
*Data Collection & Pre-processing*
```sh
raw_mail_data = pd.read_csv('mail_data.csv')
mail_data = raw_mail_data.where(pd.notnull(raw_mail_data), '')
mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1
X = mail_data['Message']
Y = mail_data['Category']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)
```
*Feature Extraction*
```sh
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')
```
*Training the Model*
```sh
model = LogisticRegression()
model.fit(X_train_features, Y_train)
```
*Evaluating the Model*
```sh
prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)
print('Accuracy on training data: ', accuracy_on_training_data)

prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)
print('Accuracy on test data: ', accuracy_on_test_data)
```
*Building a Predictive System*
```sh
input_mail = ["Just forced myself to eat a slice. I'm really not hungry tho. This sucks. Mark is getting worried. He knows I'm sick when I turn down pizza. Lol "]
input_data_features = feature_extraction.transform(input_mail)
prediction = model.predict(input_data_features)
if prediction[0] == 1:
    print('Ham mail')
else:
    print('Spam mail')
```

## Model Evaluation
- Accuracy on Training Data: The accuracy of the model on the training data is displayed after training.
- Accuracy on Test Data: The accuracy of the model on the test data is displayed after evaluation.



