#
# File name: main.py
# Author: Danique van der Wijk
# Student number: s3989771
# Last updated: 23 December 2024
# Description: The main file of the thesis
#

import os
import csv
from preprocessing import extract_text, parse_xml
import adjdel
from svm import train_svm
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
    df = pd.read_csv(input_file)

    all_sentences = []
    for _, row in df.iterrows():
        content = row['Content']
        sentences = nltk.tokenize.sent_tokenize(content, language='dutch')
        for sentence in sentences:
            all_sentences.append({'Title': row['Title'], 'Sentence': sentence})
    
    # Save the sentences into a CSV file
    sentences_df = pd.DataFrame(all_sentences)
    sentences_df.to_csv(output_file, index=False)
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
    input_file = os.path.join('data', 'nlwiki-20241220-pages-articles-multistream1.xml-p1p134538')
    output_file = 'output.csv'
    sentence_file = 'sentences.csv'
    modified_file = 'modified_sentences.csv'

    # Check if the original XML file is already converted into a csv file
    if not os.path.exists(output_file):
        process_pages(input_file, output_file)
        print("processing of pages done")
    
    # Split the sentences if that has not been done yet
    if not os.path.exists(sentence_file):
        split_into_sentences(output_file, sentence_file)
        print("Sentence splitting done")

    input_text = "test"
    bitcode = text_to_bitcode(input_text)
    print(bitcode)

    # Apply adjective deletion, if that has not been done yet
    #if not os.path.exists(modified_file):
    adjdel.bitcode_to_text(sentence_file, bitcode, modified_file, limit=100)
    print("Adjective deletion done")

    # die dingen moet ik nog labels geven voordat ik de SVM kan doen

    #model, accuracy = train_svm(modified_file)
    #print(f"SVM model is trained with an accuracy of {accuracy * 100:.2f}%")
        

if __name__ == "__main__":
    main()