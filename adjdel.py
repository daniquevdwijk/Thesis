#
# File name: adjdel.py
# Author: Danique van der Wijk
# Student number: s3989771
# Last updated: 8 January 2025
# Description: The file where the adjective deletion takes place
#

import spacy
import pandas as pd
from tqdm import tqdm


# Loads the Dutch language model of spaCy
nlp = spacy.load("nl_core_news_sm")

def count_adjectives(texts, nlp):
    """ """
    counts = []
    for doc in tqdm(nlp.pipe(texts, batch_size=100), total=len(texts), desc="Processing texts"):
        adjective_count = sum(1 for token in doc if token.pos_ == "ADJ")
        counts.append(adjective_count)
    return counts


def adjective_labeling(input_csv, output_csv):
    """ """
    # Load the input CSV
    df = pd.read_csv(input_csv)
    df = df.astype(str)

    # Check if there is a 'Content' column
    if 'Content' not in df.columns:
        raise ValueError("Input CSV must contain a 'Content' column.")
    
    # Limit the number of pages to 1000
    df = df.head(1000)
    
    # Count adjectives in every page
    df['Adjective_Count'] = count_adjectives(df['Content'], nlp)
    
    # Calculate the total number of adjectives
    total_adjectives = df['Adjective_Count'].sum()
    print(f"Total number of adjectives: {total_adjectives}")

    # Save the DataFrame to a CSV file
    df.to_csv(output_csv, index=False)
    print(f"Adjectives saved to '{output_csv}'")

    return

def find_matching_count(df, bitcode_length):
    """ Returns text with the same amount of adjectives as the length of the bitcode."""
    df = pd.read_csv(df)
    matching_texts = []
    for index, row in df.iterrows():
        if row['Adjective_Count'] == bitcode_length:
            matching_texts.append(row['Content'])
    if not matching_texts:
        print("No texts found with the same amount of adjectives as the length of the bitcode.")
    return matching_texts


def bitcode_to_text(text, bitcode):
    """ """
    doc = nlp(text)
    new_text = []
    bit_index = 0

    for token in doc:
        if token.pos_ == "ADJ": # Only the adjectives are to be manipulated
            if bit_index < len(bitcode): # Check if there is a bit left
                if bitcode[bit_index] == "1":
                    new_text.append(token.text) # Add the adjective to the new text
                # else: don't add the adjective to the new text
                bit_index += 1 # Go to the next bit
            else:
                # If there are no bits left, add the adjective to the new text
                continue
        else:
            new_text.append(token.text) # Other tokens are added to the new text
    
    # Join the tokens to form a new text
    modified_text = " ".join(new_text)
    return modified_text

    