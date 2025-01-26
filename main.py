#
# File name: main.py
# Author: Danique van der Wijk
# Student number: s3989771
# Last updated: 10 January 2025
# Description: The main file of the thesis
#

import os
import csv
from transformers import pipeline
from preprocessing import extract_text, parse_xml
from adjdel import bitcode_to_text_spacy, bitcode_to_text_stanza, adjective_labeling_spacy, adjective_labeling_stanza, find_matching_count, generate_bitcode_spacy, generate_bitcode_stanza
#from labelling import label_sentences
from evaluation import train_svm, calculate_bleu_score
import xml.etree.ElementTree as ET
import pandas as pd
import nltk

#Download NLTK tokenizer (only needed 1 time)
#nltk.download('punkt')
#nltk.download('punkt_tab')


def process_pages(input_file, output_file):
    """ """
    pages = parse_xml(input_file)
    print(f"Number of pages found: {len(pages)}") # deze kan ooit wel een keer weg
    namespace = {"ns": "http://www.mediawiki.org/xml/export-0.11/"}

    # open the csv-file to write in it
    with open(output_file, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Title', 'Content']) #column titles

        # iterate through the page elements
        for page in pages:
            title = page.find("ns:title", namespace)
            title_text = title.text if title is not None else "Untitled"

            context = extract_text(page)
            writer.writerow([title_text, context])
    
    print(f"The processed data is saved in {output_file}")


def split_into_sentences(input_file, output_file):
    """ Splits the content column of the CSV into sentences. """
    df = pd.read_csv(input_file, encoding='utf-8')

    if 'Content' not in df.columns:
        print("The input file must contain a 'Content' column.")
        return
    
    df['Content'] = df['Content'].fillna("").astype(str) # replace NaN with empty string

    all_sentences = []
    for _, row in df.iterrows():
        content = row['Content']

        # Check if the content is a string
        if not isinstance(content, str):
            continue

        sentences = nltk.tokenize.sent_tokenize(content, language='dutch')
        for sentence in sentences:
            all_sentences.append({'Sentence': sentence})
    
    # Save the sentences into a CSV file
    sentences_df = pd.DataFrame(all_sentences)
    sentences_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Sentences saved in {output_file}")


def inspect_xml_structure(input_file):
    """ """
    tree = ET.parse(input_file)
    root = tree.getroot()

    print("Root tag:", root.tag)
    #print("Child tags of root:")
    #for child in root:
        #print(" -", child.tag)


def text_to_bitcode(input_text):
    """ Convert input text to binary bitcode."""
    return ''.join(format(ord(char), '08b') for char in input_text)

def bitcode_transform(bitcode):
    """ Convert binary bitcode to text. """
    # Split the bitcode into chunks of 8 bits (1 byte)
    byte_chunks = [bitcode[i:i + 8] for i in range(0, len(bitcode), 8)]

    # Convert each 8-bit chunk to a character
    text = ''.join(chr(int(byte, 2)) for byte in byte_chunks)

    return text


def debug_bitcode(original_bitcode, reconstructed_bitcode):
    """" """
    for i, (original_bit, reconstructed_bit) in enumerate(zip(original_bitcode, reconstructed_bitcode)):
        if original_bit != reconstructed_bit:
            print(f"Bit mismatch at position {i}: Original = {original_bit}, Reconstructed = {reconstructed_bit}")
    
    #assert original_bitcode == reconstructed_bitcode, "The original bitcode and the reconstructed bitcode are not the same."


def main():
    """ """
    input_file = os.path.join('data', 'nlwiki-20241220-pages-articles-multistream1.xml-p1p134538')
    output_file = 'output.csv'
    count_file_spacy = 'count_spacy.csv'
    count_file_stanza = 'count_stanza.csv'

    input_text = "Test Test Test"
    print(f"Original text: {input_text}")
    bitcode = text_to_bitcode(input_text)
    print(f"Original bitcode: {bitcode}")
    #print(f"The lenght of the bitcode is: {len(bitcode)} bits")

    # Check if the original XML file is already converted into a csv file
    if not os.path.exists(output_file):
        process_pages(input_file, output_file)
        print("processing of pages done")

    if not os.path.exists(count_file_spacy):
        adjective_labeling_spacy(output_file, count_file_spacy)
        print("Adjective labeling done (spacy)")
    
    if not os.path.exists(count_file_stanza):
        adjective_labeling_stanza(output_file, count_file_stanza)
        print("Adjective labeling done (stanza)")

    cover_text = find_matching_count(count_file_spacy, len(bitcode))
    if cover_text:
        print("Matching text found")
        print(cover_text)
    else:
        print("No matching text found")
    
    stego_text = bitcode_to_text_spacy(cover_text[0], bitcode)
    #stego_text = bitcode_to_text_stanza(cover_text[0], bitcode)
    print(f"This is the stego text: {stego_text}")

    cover_bitcode = generate_bitcode_spacy(cover_text[0], stego_text)
    #cover_bitcode = generate_bitcode_stanza(cover_text[0], stego_text)
    print(f"This is the bitcode decoded from the stego text: {cover_bitcode}")
    #assert bitcode == cover_bitcode, "The original bitcode and the stego bitcode are not the same."

    debug_bitcode(bitcode, cover_bitcode)
    covered_text = bitcode_transform(cover_bitcode)
    print(f"This is the hidden text: {covered_text}")

    #assert input_text == covered_text, "The original text and the hidden text are not the same."


    # die dingen moet ik nog labels geven voordat ik de SVM kan doen

    #model, accuracy = train_svm(labeled_file)
    #print(f"SVM model is trained with an accuracy of {accuracy * 100:.2f}%")

    bleu_score = calculate_bleu_score(stego_text, cover_text[0])
    

if __name__ == "__main__":
    main()