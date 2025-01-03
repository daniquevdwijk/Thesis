#
# File name: svm.py
# Author: Danique van der Wijk
# Student number: s3989771
# Last updated: 30 December 2024
# Description: The file with the code to execute the Support Vector Machine (SVM)
#

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pandas as pd

def train_svm(csv_file):
    """ """
    # load data
    data = pd.read_csv(csv_file)
    X = data.iloc[:, :-1] # Features
    y = data.iloc[:, -1] # Labels

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 23)

    # Make SVM model and train it
    model = SVC(kernel = 'linear')
    model.fit(x_train, y_train)

    # Accuracy
    accuracy = model.score(x_test, y_test)
    
    return model, accuracy
