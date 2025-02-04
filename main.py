#
# File name: main.py
# Author: Danique van der Wijk
# Student number: s3989771
# Last updated: 3 February 2025
# Description: The main file of the thesis
#

import os
import csv
from preprocessing import extract_text, parse_xml
from adjdel import bitcode_to_text_spacy, bitcode_to_text_stanza, adjective_labeling_spacy, adjective_labeling_stanza, find_matching_count, generate_bitcode_spacy, generate_bitcode_stanza
from evaluation import calculate_bleu_score, perplexity, levenshtein_dist


def process_pages(input_file, output_file):
    """
    Processes pages from an XML input file and writes the extracted data to a CSV output file.
    Args:
        input_file (str): The path to the XML file containing the pages to be processed.
        output_file (str): The path to the CSV file where the processed data will be saved.
    Returns:
        None
    The function performs the following steps:
    1. Parses the XML input file to extract page elements.
    2. Prints the number of pages found.
    3. Opens the CSV output file for writing.
    4. Writes the column titles 'Title' and 'Content' to the CSV file.
    5. Iterates through each page element, extracting the title and content.
    6. Writes the extracted title and content to the CSV file.
    7. Prints a message indicating that the processed data has been saved.
    """
    pages = parse_xml(input_file)
    print(f"Number of pages found: {len(pages)}")
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


def text_to_bitcode(input_text):
    """
    Convert input text to binary bitcode.

    Args:
        input_text (str): The text to be converted to binary.

    Returns:
        str: A string representing the binary bitcode of the input text.
    """
    return ''.join(format(ord(char), '08b') for char in input_text)

def bitcode_transform(bitcode):
    """
    Convert binary bitcode to text.
    Args:
        bitcode (str): A string of binary digits (0s and 1s) representing the bitcode.
    Returns:
        str: The decoded text obtained by converting each 8-bit chunk of the bitcode to its corresponding character.
    """
    # Split the bitcode into chunks of 8 bits (1 byte)
    byte_chunks = [bitcode[i:i + 8] for i in range(0, len(bitcode), 8)]

    # Convert each 8-bit chunk to a character
    text = ''.join(chr(int(byte, 2)) for byte in byte_chunks)

    return text



def evaluate_sentences(sentences, stanza_file, spacy_file):
    """
    Evaluates a list of sentences by encoding them into bitcodes and then embedding these bitcodes into cover texts
    using both spaCy and Stanza methods. The function calculates various metrics for each sentence and returns the results.
    Args:
        sentences (list of str): List of sentences to be evaluated.
        stanza_file (str): Path to the file containing cover texts for Stanza.
        spacy_file (str): Path to the file containing cover texts for spaCy.
    Returns:
        list of dict: A list of dictionaries containing the evaluation results for each sentence. Each dictionary contains:
            - "sentence" (str): The original sentence.
            - "cover_text_spacy" (str): The cover text used for spaCy.
            - "cover_text_stanza" (str): The cover text used for Stanza.
            - "stego_text_spacy" (str): The stego text generated using spaCy.
            - "stego_text_stanza" (str): The stego text generated using Stanza.
            - "decoded_spacy" (str): The decoded bitcode from the stego text using spaCy.
            - "decoded_stanza" (str): The decoded bitcode from the stego text using Stanza.
            - "bleu_spacy" (float): The BLEU score for the spaCy stego text.
            - "bleu_stanza" (float): The BLEU score for the Stanza stego text.
            - "perplexity_spacy_cover" (float): The perplexity of the spaCy cover text.
            - "perplexity_spacy_stego" (float): The perplexity of the spaCy stego text.
            - "perplexity_stanza_cover" (float): The perplexity of the Stanza cover text.
            - "perplexity_stanza_stego" (float): The perplexity of the Stanza stego text.
            - "lev_dist_spacy" (int): The Levenshtein distance between the original sentence and the decoded bitcode using spaCy.
            - "lev_dist_stanza" (int): The Levenshtein distance between the original sentence and the decoded bitcode using Stanza.
    """
    results = []
    for sentence in sentences:
        bitcode = text_to_bitcode(sentence)
        print()
        print(sentence)

        # SpaCy
        # Find a fitting covertext
        cover_text_spacy = find_matching_count(spacy_file, len(bitcode))
        if not cover_text_spacy:
            print(f"No matching text found for: {sentence} (spaCy)")
            continue
        cover_text_spacy = cover_text_spacy[0]
        stego_text_spacy = bitcode_to_text_spacy(cover_text_spacy, bitcode)
        stego_bitcode_spacy = generate_bitcode_spacy(cover_text_spacy, stego_text_spacy)
        covered_bitcode_spacy = bitcode_transform(stego_bitcode_spacy)
        bleu_spacy = calculate_bleu_score(stego_text_spacy, cover_text_spacy, (0.25,0.25,0.25,0.25))
        perp_cover_spacy = perplexity(cover_text_spacy)
        perp_stego_spacy = perplexity(stego_text_spacy)
        lev_dist_spacy = levenshtein_dist(sentence, covered_bitcode_spacy)

        # Stanza
        # Find a fitting covertext
        cover_text_stanza = find_matching_count(stanza_file, len(bitcode))
        if not cover_text_stanza:
            print(f"No matching text found for: {sentence} (Stanza)")
            continue
        cover_text_stanza = cover_text_stanza[0]
        stego_text_stanza = bitcode_to_text_stanza(cover_text_stanza, bitcode)
        stego_bitcode_stanza = generate_bitcode_stanza(cover_text_stanza, stego_text_stanza)
        covered_bitcode_stanza = bitcode_transform(stego_bitcode_stanza)
        bleu_stanza = calculate_bleu_score(stego_text_stanza, cover_text_stanza, (0.25,0.25,0.25,0.25))
        perp_cover_stanza = perplexity(cover_text_stanza)
        perp_stego_stanza = perplexity(stego_text_stanza)
        lev_dist_stanza = levenshtein_dist(sentence, covered_bitcode_stanza)

        results.append({
            "sentence": sentence,
            "cover_text_spacy": cover_text_spacy,
            "cover_text_stanza": cover_text_stanza,
            "stego_text_spacy": stego_text_spacy,
            "stego_text_stanza": stego_text_stanza,
            "decoded_spacy": covered_bitcode_spacy,
            "decoded_stanza": covered_bitcode_stanza,
            "bleu_spacy": bleu_spacy,
            "bleu_stanza": bleu_stanza,
            "perplexity_spacy_cover": perp_cover_spacy,
            "perplexity_spacy_stego": perp_stego_spacy,
            "perplexity_stanza_cover": perp_cover_stanza,
            "perplexity_stanza_stego": perp_stego_stanza,
            "lev_dist_spacy": lev_dist_spacy,
            "lev_dist_stanza": lev_dist_stanza
        })

    return results    


def main():
    """ 
    Main function to process the Wikimedia data, extract the adjectives
    and evaluate sample sentences.

    Output:
    - 'output.csv': processed Wikimedia data in a CSV file.
    - 'count_spacy.csv': adjective labeling results using SpaCy.
    - 'count_stanza.csv': adjective labeling results using Stanza.
    - Console output with evaluation metrics for the sample sentences.
    """

    input_file = os.path.join('data', 'nlwiki-20241220-pages-articles-multistream1.xml-p1p134538')
    output_file = 'output.csv'
    count_file_spacy = 'count_spacy.csv'
    count_file_stanza = 'count_stanza.csv'

    # Check if the original XML file is already converted into a csv file
    if not os.path.exists(output_file):
        process_pages(input_file, output_file)
        print("Processing of pages done")

    # Check if there is already a file with adjective count done by SpaCy
    if not os.path.exists(count_file_spacy):
        adjective_labeling_spacy(output_file, count_file_spacy)
        print("Adjective labeling done (spacy)")

    # Check if there is already a file with adjective count done by Stanza
    if not os.path.exists(count_file_stanza):
        adjective_labeling_stanza(output_file, count_file_stanza)
        print("Adjective labeling done (stanza)")

    # Function to count adjectives from a CSV file
    def count_adjectives(file_path):
        with open(file_path, mode='r', encoding='utf-8') as csv_file:
            reader = csv.DictReader(csv_file)
            return sum(int(row['Adjective_Count']) for row in reader)

    # Print the number of adjectives found by SpaCy
    total_adjectives_spacy = count_adjectives(count_file_spacy)
    print(f"Total adjectives found by SpaCy: {total_adjectives_spacy}")

    # Print the number of adjectives found by Stanza
    total_adjectives_stanza = count_adjectives(count_file_stanza)
    print(f"Total adjectives found by Stanza: {total_adjectives_stanza}")

    # Test sentences
    sentences = [
        "De kat zit op de paal",
        "Mijn fiets is stuk",
        "Het is mooi weer",
        "Ik drink koffie",
        "De zon is fel",
        "Het huis heeft een deur",
        "We gaan naar de kroeg",
        "Mijn hond blaft",
        "De mok staat op tafel",
        "Ik houd van thee"
    ]

    # Generate results
    results = evaluate_sentences(sentences,count_file_stanza, count_file_spacy)

    # Print results
    for result in results:
        print(f"Original sentence: {result['sentence']}")
        #print(f"Selected Cover text with spaCy: {result['cover_text_spacy']}")
        #print(f"Selected Cover text with Stanza: {result['cover_text_stanza']}")
        #print(f"Stego text with spaCy: {result['stego_text_spacy']}")
        #print(f"Stego text with Stanza: {result['stego_text_stanza']}")
        print(f"Decoded message (spaCy): {result['decoded_spacy']}")
        print(f"Decoded message (Stanza): {result['decoded_stanza']}")
        print(f"BLEU score (spaCy): {result['bleu_spacy']:.4f}")
        print(f"BLEU score (Stanza) {result['bleu_stanza']:.4f}")
        print(f"Perplexity Score Cover (spaCy): {result['perplexity_spacy_cover']:.2f}")
        print(f"Perplexity Score Stego (spaCy): {result['perplexity_spacy_stego']:.2f}")
        print(f"Perplexity Score Cover (Stanza): {result['perplexity_stanza_cover']:.2f}")
        print(f"Perplexity Score Stego (Stanza): {result['perplexity_stanza_stego']:.2f}")
        print(f"Levenshtein Distance (spaCy): {result['lev_dist_spacy']}")
        print(f"Levenshtein Distance (Stanza): {result['lev_dist_stanza']}")
        print("-" * 50) # For readability


if __name__ == "__main__":
    main()