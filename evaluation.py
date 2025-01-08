#
# File name: evaluation.py
# Author: Danique van der Wijk
# Student number: s3989771
# Last updated: 8 January 2025
# Description: The file with the code to execute the Support Vector Machine (SVM) and the calculation of the BLEU scores
#


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from nltk.translate.bleu_score import sentence_bleu
import pandas as pd

def train_svm(input_file):
    """ """
    # load data
    data = pd.read_csv(input_file)
    print(f'Columns: {data.columns.tolist()}')

    if 'Sentence' not in data.columns or 'Naturalness' not in data.columns:
        raise ValueError("Input CSV must contain 'Sentence' and 'Naturalness' columns.")

    X = data['Sentence'] # Features
    y = data['Naturalness'] # Labels

    # Convert text to numerical representation using TF-IDF
    vectorizer = TfidfVectorizer()
    X_transformed = vectorizer.fit_transform(X)


    # Split data
    x_train, x_test, y_train, y_test = train_test_split(X_transformed, y, test_size = 0.2, random_state = 23)

    # Make SVM model and train it
    model = SVC(kernel = 'linear')
    model.fit(x_train, y_train)

    # Evaluate model
    y_pred = model.predict(x_test)
    accuracy = model.score(x_test, y_pred)
    
    return model, accuracy

def calculate_bleu_score(input_file):
    """ """
    data = pd.read_csv(input_file)
    print(f'Columns: {data.columns.tolist()}')

    if 'Sentence' not in data.columns or 'Modified Sentence' not in data.columns:
        raise ValueError("Input CSV must contain 'Sentence' and 'Modified Sentence' columns.")

    bleu_scores = []
    for _, row in data.iterrows():
        sentence = row['Sentence']
        modified_sentence = row['Modified Sentence']

        # Calculate BLEU score
        reference = sentence.split()
        candidate = modified_sentence.split()
        score = sentence_bleu([reference], candidate)
        bleu_scores.append(score)

    # Add BLEU scores to a DataFrame
    data['BLEU Score'] = bleu_scores

    # Calculate average BLEU score
    avg_bleu_score = sum(bleu_scores) / len(bleu_scores)
    print(f"Average BLEU score: {avg_bleu_score}")

