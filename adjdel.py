#
# File name: adjdel.py
# Author: Danique van der Wijk
# Student number: s3989771
# Last updated: 8 January 2025
# Description: The file where the adjective deletion takes place
#

import spacy
import pandas as pd


# Loads the Dutch language model of spaCy
nlp = spacy.load("nl_core_news_sm")


def bitcode_to_text(input_csv, bitcode, output_file, limit=512):
    """ """
    # Load the input CSV
    df = pd.read_csv(input_csv)
    df = df.astype(str)

    # Check if there is a 'Sentence' column
    if 'Sentence' not in df.columns:
        raise ValueError("Input CSV must contain a 'Sentence' column.")
    
    # make a list of the sentences
    df_limited = df.head(limit)
    sentences = df_limited['Sentence'].tolist()

    modified_sentences = []
    bit_index = 0 # keeps which bit from the bitcode is used

    for sentence in sentences:
        # parse the sentence with spaCy
        doc = nlp(sentence)
        # checks if there are adjectives in the sentence
        adjectives = [token for token in doc if token.pos_ == "ADJ"]

        if adjectives:
            # only applies if there are adjectives
            if bit_index < len(bitcode):
                bit = bitcode[bit_index] # get the current bit
                bit_index += 1 

                if bit == "1":
                    # Keep adjectives in sentence
                    modified_sentences.append(sentence)
                else:
                    # Delete adjectives
                    cleaned_sentence = " ".join([token.text for token in doc if token.pos_ != "ADJ"])
                    modified_sentences.append(cleaned_sentence)
            else:
                # if the bitcode is empty, add original sentence
                modified_sentences.append(sentence)
        else:
            # add sentences without adjectives unchanged
            modified_sentences.append(sentence)
    
    # Create DataFrame with original and modified sentences
    df_limited.loc[:, 'Modified Sentence'] = modified_sentences

    # Save the Dataframe to a CSV file
    df_limited.to_csv(output_file, index=False)
    print(f"Modified sentences saved to '{output_file}'")

    return