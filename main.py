#
# File name: main.py
# Author: Danique van der Wijk
# Student number: s3989771
# Last updated: 15 December 2024
# Description: The main file of the thesis
#

import os
import csv
from preprocessing import extract_text, parse_xml
import xml.etree.ElementTree as ET


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


def inspect_xml_structure(input_file):
    """ """
    tree = ET.parse(input_file)
    root = tree.getroot()

    print("Root tag:", root.tag)
    print("Child tags of root:")
    #for child in root:
        #print(" -", child.tag)


def main():
    """ """
    input_file = os.path.join('data', 'nlwiki-20241201-pages-articles1.xml-p1p134538')
    output_file = 'output.csv'
    
    # check if the input file is present
    if not os.path.exists(input_file):
        print(f"Inputfile not found: {input_file}")
    else:
        #inspect_xml_structure(input_file)
        process_pages(input_file, output_file)



if __name__ == "__main__":
    main()