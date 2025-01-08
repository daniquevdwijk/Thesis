#
# File name: main.py
# Author: Danique van der Wijk
# Student number: s3989771
# Last updated: 8 January 2025
# Description: The main file of the thesis
#

import os
import csv
from preprocessing import extract_text, parse_xml
import adjdel
from labelling import label_sentences
from evaluation import train_svm
from evaluation import calculate_bleu_score
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
        writer.writerow(['Content']) #column titles

        # iterate through the page elements
        for page in pages:
            context = extract_text(page)
            writer.writerow([context])
    
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
    """ """
    return ''.join(format(ord(char), '08b') for char in input_text)


def main():
    """ """
    input_file = os.path.join('data', 'nlwiki-20241201-pages-articles1.xml-p1p134538')
    output_file = 'output.csv'
    sentence_file = 'sentences.csv'
    modified_file = 'modified_sentences.csv'
    labeled_file = 'labeled_sentences.csv'

    # Check if the original XML file is already converted into a csv file
    if not os.path.exists(output_file):
        process_pages(input_file, output_file)
        print("processing of pages done")
    
    # Split the sentences if that has not been done yet
    if not os.path.exists(sentence_file):
        split_into_sentences(output_file, sentence_file)
        print("Sentence splitting done")

    input_text = "Dit is een testzin."
    bitcode = text_to_bitcode(input_text)
    print(bitcode)

    # Apply adjective deletion, if that has not been done yet
    if not os.path.exists(modified_file):
        adjdel.bitcode_to_text(sentence_file, bitcode, modified_file, limit=512)
        print("Adjective deletion done")

    if not os.path.exists(labeled_file):
        label_sentences(modified_file, labeled_file)

    # die dingen moet ik nog labels geven voordat ik de SVM kan doen

    model, accuracy = train_svm(labeled_file)
    print(f"SVM model is trained with an accuracy of {accuracy * 100:.2f}%")

    calculate_bleu_score(labeled_file)
    

if __name__ == "__main__":
    main()