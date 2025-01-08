#
# File name: labelling.py
# Author: Danique van der Wijk
# Student number: s3989771
# Last updated: 3 January 2025
# Description: This code labels the sentences
#

import pandas as pd
from transformers import pipeline


def is_natural(sentence, classifier):
    """ """
    try:
        result = classifier(sentence)[0] # classify the sentence
        if result['label'] == 'LABEL_1' and result['score'] > 0.5:
            return "Natural"
        else:
            return "Unnatural"
    except Exception as e:
        return "Error"


def label_sentences(input_file, output_file, model_name="GroNLP/bert-base-dutch-cased"):
    """ """
    try:
        # Load dataset
        df = pd.read_csv(input_file)

        # Check if 'Modified Sentence column exists
        if "Modified Sentence" not in df.columns:
            raise ValueError("The input file must contain a 'Modified Sentence' column.")
        
        # Drops rows with missing sentences
        df = df.dropna(subset=["Modified Sentence"])

        # the pre-trained pipeline
        classifier = pipeline("text-classification", model=model_name)

        # apply the classification
        df["Naturalness"] = df["Modified Sentence"].apply(lambda x: is_natural(x, classifier))

        # safe the labelled dataset
        df.to_csv(output_file, index=False)
        print(f"Labeled dataset saved to {output_file}")

    except Exception as e:
        print(f"An error occured: {e}")
