#
# File name: adjdel.py
# Author: Danique van der Wijk
# Student number: s3989771
# Last updated: 8 January 2025
# Description: The file where the adjective deletion takes place
#

import spacy
import stanza
#from transformers import pipeline
#import nltk
#from nltk import pos_tag
#from nltk.tokenize import word_tokenize
#from nltk.corpus import alpino as dutch_corpus
#from pattern.nl import tag
import pandas as pd
from tqdm import tqdm

# Only needed one time to download the Dutch model of Stanza
stanza.download("nl")

# Download NLTK tokenizer (only needed 1 time)
#nltk.download('punkt')
#nltk.download('alpino')
#nltk.download('averaged_perceptron_tagger_eng')

#pos_tagger = pipeline("token-classification", model="GroNLP/bert-base-dutch-cased")

# Initialize the Stanza pipeline for Dutch
nlp_stanza = stanza.Pipeline(lang="nl", processors="tokenize,pos")

nlp_spacy = spacy.load("nl_core_news_sm")

def count_adjectives_spacy(texts):
    """ """
    counts = []
    for text in tqdm(texts, total=len(texts), desc="Processing texts"):
        doc = nlp_spacy(text)
        adjective_count = sum(1 for token in doc if token.pos_ == "ADJ")
        counts.append(adjective_count)
    return counts

def count_adjectives_stanza(texts):
    """ """
    counts = []
    for text in tqdm(texts, total=len(texts), desc="Processing texts"):
        doc = nlp_stanza(text)
        adjective_count = sum(1 for sent in doc.sentences for word in sent.words if word.upos == "ADJ")
        counts.append(adjective_count)
    return counts


def adjective_labeling_spacy(input_csv, output_csv):
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
    df['Adjective_Count'] = count_adjectives_spacy(df['Content'])
    
    # Calculate the total number of adjectives
    total_adjectives = df['Adjective_Count'].sum()
    print(f"Total number of adjectives: {total_adjectives}")

    # Save the DataFrame to a CSV file
    df.to_csv(output_csv, index=False)
    print(f"Adjectives saved to '{output_csv}'")

    return

def adjective_labeling_stanza(input_csv, output_csv):
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
    df['Adjective_Count'] = count_adjectives_stanza(df['Content'])
    
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


def bitcode_to_text_spacy(text, bitcode):
    """ """
    doc = nlp_spacy(text)
    new_text = []
    bit_index = 0

    print("Cover adjectives:", [token.text for token in doc if token.pos_ == "ADJ"])

    for token in doc:
        if token.pos_ == "ADJ": # Only manipulate adjectives
            if bit_index < len(bitcode): # Check if there is a bit left
                if bitcode[bit_index] == "1": 
                    new_text.append(token.text) # Add the adjective in the new text
                    print(f"Bit: {bitcode[bit_index]}, Adjective added: {token.text}")
                # Else: skip the adjective
                else:
                    print(f"Bit: {bitcode[bit_index]}, Adjective skipped: {token.text}")
                bit_index += 1 # Go to the next bit
            # Else: skip the adjective
        else:
            new_text.append(token.text) # Other tokens added to the new text
    
    # Join the tokens to form a new text
    modified_text = " ".join(new_text)
    return modified_text


def bitcode_to_text_stanza(text, bitcode):
    """ """
    doc = nlp_stanza(text)
    new_text = []
    bit_index = 0

    print("Cover adjectives:", [word.text for sent in doc.sentences for word in sent.words if word.upos == "ADJ"])

    for sent in doc.sentences:
        for word in sent.words:
            if word.upos == "ADJ": # Only manipulate adjectives
                if bit_index < len(bitcode): # Check if there is a bit left
                    if bitcode[bit_index] == "1": 
                        new_text.append(word.text) # Add the adjective in the new text
                        print(f"Bit: {bitcode[bit_index]}, Adjective added: {word.text}")
                    # Else: skip the adjective
                    else:
                        print(f"Bit: {bitcode[bit_index]}, Adjective skipped: {word.text}")
                    bit_index += 1 # Go to the next bit
                # Else: skip the adjective
            else:
                new_text.append(word.text) # Other tokens added to the new text
    
    # Join the tokens to form a new text
    modified_text = " ".join(new_text)
    return modified_text


def generate_bitcode_spacy(cover_text, stego_text):
    """ """
    cover_doc = nlp_spacy(cover_text)
    stego_doc = nlp_spacy(stego_text)

    # Print POS tags for cover and stego texts
    print("Cover text POS tags:", [(token.text, token.pos_) for token in cover_doc])
    print()
    print("Stego text POS tags:", [(token.text, token.pos_) for token in stego_doc])

    # List of all tokens in stego text
    cover_adj = [token.text for token in cover_doc if token.pos_ == "ADJ"]
    stego_adj = [token.text for token in stego_doc if token.pos_ == "ADJ"]

    # Debugging:
    #print("Cover text tokens:")
    #for sent in cover_doc.sentences:
        #print([(word.text, word.upos) for word in sent.words])
    #print("Stego text tokens:")
    #for sent in stego_doc.sentences:
        #print([(word.text, word.upos) for word in sent.words])
    
    bitcode = []
    stego_index = 0

    for adj in cover_adj:
        # Check if there are adjectives in stego to compare
        if stego_index < len(stego_adj) and adj == stego_adj[stego_index]:
            bitcode.append("1")
            stego_index += 1
            #print("Adjective found in stego:", adj)
        else:
            bitcode.append("0")
    
    # Ensure bitcode matches length of cover_adj
    while len(bitcode) < len(cover_adj):
        bitcode.append("0")
    
    #Debugging:
    if len(bitcode) != len(cover_adj):
        print("Bitcode length does not match the number of adjectives in the cover")
    
    #print("Cover adjectives:", cover_adj)
    #print("Stego adjectives:", stego_adj)
    #print("Cover bitcode:", bitcode)

    return "".join(bitcode)

def generate_bitcode_stanza(cover_text, stego_text):
    """ """
    cover_doc = nlp_stanza(cover_text)
    stego_doc = nlp_stanza(stego_text)

    print("Cover text POS tags:", [(word.text, word.upos) for sent in cover_doc.sentences for word in sent.words])
    print()
    print("Stego text POS tags:", [(word.text, word.upos) for sent in stego_doc.sentences for word in sent.words])

    # List of all tokens in stego text
    cover_adj = [word.text for sent in cover_doc.sentences for word in sent.words if word.upos == "ADJ"]
    stego_adj = [word.text for sent in stego_doc.sentences for word in sent.words if word.upos == "ADJ"]

    bitcode = []
    stego_index = 0

    for adj in cover_adj:
        # Check if there are adjectives in stego to compare
        if stego_index < len(stego_adj) and adj == stego_adj[stego_index]:
            bitcode.append("1")
            stego_index += 1
            #print("Adjective found in stego:", adj)
        else:
            bitcode.append("0")
    
    # Ensure bitcode matches length of cover_adj
    while len(bitcode) < len(cover_adj):
        bitcode.append("0")
    
    #Debugging:
    if len(bitcode) != len(cover_adj):
        print("Bitcode length does not match the number of adjectives in the cover")
    
    #print("Cover adjectives:", cover_adj)
    #print("Stego adjectives:", stego_adj)
    #print("Cover bitcode:", bitcode)

    return "".join(bitcode)