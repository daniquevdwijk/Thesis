#
# File name: adjdel.py
# Author: Danique van der Wijk
# Student number: s3989771
# Last updated: 3 February 2025
# Description: The file where the adjective deletion takes place
#

import spacy
import stanza
import pandas as pd
from tqdm import tqdm

# Only needed one time to download the Dutch model of Stanza
stanza.download("nl")

# Initialize the Stanza pipeline for Dutch
nlp_stanza = stanza.Pipeline(lang="nl", processors="tokenize,pos")

nlp_spacy = spacy.load("nl_core_news_sm")

def count_adjectives_spacy(texts):
    """
    Counts the number of adjectives in each text using spaCy.

    Args:
        texts (list of str): A list of texts to process.

    Returns:
        list of int: A list containing the count of adjectives for each text.
    """
    counts = []
    for text in tqdm(texts, total=len(texts), desc="Processing texts"):
        doc = nlp_spacy(text)
        adjective_count = sum(1 for token in doc if token.pos_ == "ADJ")
        counts.append(adjective_count)
    return counts

def count_adjectives_stanza(texts):
    """
    Counts the number of adjectives in each text using Stanza.

    Args:
        texts (list of str): A list of texts to process.

    Returns:
        list of int: A list containing the count of adjectives for each text.
    """
    counts = []
    for text in tqdm(texts, total=len(texts), desc="Processing texts"):
        doc = nlp_stanza(text)
        adjective_count = sum(1 for sent in doc.sentences for word in sent.words if word.upos == "ADJ")
        counts.append(adjective_count)
    return counts


def adjective_labeling_spacy(input_csv, output_csv):
    """
    Processes an input CSV file to count adjectives in the 'Content' column using spaCy,
    and saves the results to an output CSV file.
    Args:
        input_csv (str): Path to the input CSV file containing a 'Content' column.
        output_csv (str): Path to the output CSV file where results will be saved.
    Raises:
        ValueError: If the input CSV does not contain a 'Content' column.
    Returns:
        None
    """
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
    #print(f"Total number of adjectives: {total_adjectives}")

    # Save the DataFrame to a CSV file
    df.to_csv(output_csv, index=False)
    print(f"Adjectives count saved to '{output_csv}'")

    return

def adjective_labeling_stanza(input_csv, output_csv):
    """
    Processes an input CSV file to count adjectives in the 'Content' column using the Stanza library,
    and saves the results to an output CSV file.
    Args:
        input_csv (str): Path to the input CSV file containing a 'Content' column.
        output_csv (str): Path to the output CSV file where results will be saved.
    Raises:
        ValueError: If the input CSV does not contain a 'Content' column.
    Returns:
        None
    """
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
    """
    Returns a list of texts from a DataFrame that have the same number of adjectives as the specified bitcode length.

    Args:
        df (str): The file path to the CSV file containing the data.
        bitcode_length (int): The length of the bitcode, which corresponds to the number of adjectives to match.

    Returns:
        list: A list of texts (strings) from the DataFrame where the 'Adjective_Count' column matches the bitcode_length.
    """
    df = pd.read_csv(df)
    matching_texts = []
    for index, row in df.iterrows():
        if row['Adjective_Count'] == bitcode_length:
            matching_texts.append(row['Content'])
    if not matching_texts:
        print("No texts found with the same amount of adjectives as the length of the bitcode.")
    return matching_texts


def bitcode_to_text_spacy(text, bitcode):
    """
    Modify the input text by selectively including adjectives based on a bitcode.
    This function uses spaCy to tokenize the input text and processes each token.
    If a token is an adjective (ADJ) and the corresponding bit in the bitcode is '1',
    the adjective is included in the new text. Otherwise, it is skipped. All other
    tokens are included in the new text regardless of the bitcode.
    Args:
        text (str): The input text to be processed.
        bitcode (str): A string of bits ('0' and '1') indicating which adjectives to include.
    Returns:
        str: The modified text with selected adjectives included.
    """
    doc = nlp_spacy(text)
    new_text = []
    bit_index = 0

    for token in doc:
        if token.pos_ == "ADJ": # Only manipulate adjectives
            if bit_index < len(bitcode): # Check if there is a bit left
                if bitcode[bit_index] == "1": 
                    new_text.append(token.text) # Add the adjective in the new text
                    #print(f"Bit: {bitcode[bit_index]}, Adjective added: {token.text}")
                # Else: skip the adjective
                bit_index += 1 # Go to the next bit
            # Else: skip the adjective
        else:
            new_text.append(token.text) # Other tokens added to the new text
    
    # Join the tokens to form a new text
    modified_text = " ".join(new_text)
    return modified_text


def bitcode_to_text_stanza(text, bitcode):
    """ 
    Modify the input text by selectively including adjectives based on a bitcode.
    This function uses Stanza to tokenize the input text and processes each token.
    If a token is an adjective (ADJ) and the corresponding bit in the bitcode is '1',
    the adjective is included in the new text. Otherwise, it is skipped. All other
    tokens are included in the new text regardless of the bitcode.
    Args:
        text (str): The input text to be processed.
        bitcode (str): A string of bits ('0' and '1') indicating which adjectives to include.
    Returns:
        str: The modified text with selected adjectives included.
    """
    doc = nlp_stanza(text)
    new_text = []
    bit_index = 0

    for sent in doc.sentences:
        for word in sent.words:
            if word.upos == "ADJ": # Only manipulate adjectives
                if bit_index < len(bitcode): # Check if there is a bit left
                    if bitcode[bit_index] == "1": 
                        new_text.append(word.text) # Add the adjective in the new text
                    # Else: skip the adjective
                    bit_index += 1 # Go to the next bit
                # Else: skip the adjective
            else:
                new_text.append(word.text) # Other tokens added to the new text
    
    # Join the tokens to form a new text
    modified_text = " ".join(new_text)
    return modified_text


def generate_bitcode_spacy(cover_text, stego_text):
    """
    Generates a bitcode based on the comparison of adjectives in the cover text and stego text.
    The function uses spaCy to tokenize and identify adjectives in both the cover and stego texts.
    It then generates a bitcode where each bit represents whether the corresponding adjective in the
    cover text is present in the stego text at the same position.
    Args:
        cover_text (str): The original cover text.
        stego_text (str): The stego text which may contain hidden information.
    Returns:
        str: A string of bits ('0' or '1') where '1' indicates the presence of the adjective from the
             cover text in the stego text at the same position, and '0' indicates its absence.
    """
    cover_doc = nlp_spacy(cover_text)
    stego_doc = nlp_spacy(stego_text)

    # List of all tokens in stego text
    cover_adj = [token.text for token in cover_doc if token.pos_ == "ADJ"]
    stego_adj = [token.text for token in stego_doc if token.pos_ == "ADJ"]
    
    bitcode = []
    stego_index = 0

    for adj in cover_adj:
        # Check if there are adjectives in stego to compare
        if stego_index < len(stego_adj) and adj == stego_adj[stego_index]:
            bitcode.append("1")
            stego_index += 1
        else:
            bitcode.append("0")
    
    # Ensure bitcode matches length of cover_adj
    while len(bitcode) < len(cover_adj):
        bitcode.append("0")
    
    #Debugging:
    if len(bitcode) != len(cover_adj):
        print("Bitcode length does not match the number of adjectives in the cover")

    return "".join(bitcode)

def generate_bitcode_stanza(cover_text, stego_text):
    """ 
    Generates a bitcode based on the comparison of adjectives in the cover text and stego text.
    The function uses Stanza to tokenize and identify adjectives in both the cover and stego texts.
    It then generates a bitcode where each bit represents whether the corresponding adjective in the
    cover text is present in the stego text at the same position.
    Args:
        cover_text (str): The original cover text.
        stego_text (str): The stego text which may contain hidden information.
    Returns:
        str: A string of bits ('0' or '1') where '1' indicates the presence of the adjective from the
             cover text in the stego text at the same position, and '0' indicates its absence.
    """
    cover_doc = nlp_stanza(cover_text)
    stego_doc = nlp_stanza(stego_text)

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
        else:
            bitcode.append("0")
    
    # Ensure bitcode matches length of cover_adj
    while len(bitcode) < len(cover_adj):
        bitcode.append("0")
    
    #Debugging:
    if len(bitcode) != len(cover_adj):
        print("Bitcode length does not match the number of adjectives in the cover")

    return "".join(bitcode)