#
# File name: preprocessing.py
# Author: Danique van der Wijk
# Student number: s3989771
# Last updated: 15 December 2024
# Description: The file where the preprocessing of the Wikipedia dataset takes place
#

import re
import xml.etree.ElementTree as ET

def extract_text(page):
    """ """
    namespace = {"ns": "http://www.mediawiki.org/xml/export-0.11/"}

    # Find the <revision>-tag
    revision = page.find("ns:revision", namespace)
    if revision is None:
        print("No <revision> tag found")
        return ""
    
    # Find the <text>-tag within <revision>
    text_element = revision.find("ns:text", namespace)
    if text_element is None:
        print("No <text>-tag found within <revision>")
        return ""
    
    # Checks if text is empty
    if text_element.text is None:
        print("No text content in <text> tag")
        return ""
    
    # Cleans text
    raw_text = text_element.text
    cleaned_text = clean_wikitext(raw_text)

    return cleaned_text


def clean_wikitext(text):
    """ """
    if not text:
        return ""
    
    # Remove templates {{...}}
    text = re.sub(r"\{\{.*?\}\}", "", text, flags=re.DOTALL)

    # Keep content of <ref>...</ref>, but removes tags
    text = re.sub(r"<ref.*?>(.*?)</ref>", r"(\1)", text, flags=re.DOTALL)
    text = re.sub(r"<ref.*?/>", "", text)

    # Remove tables {|...|}
    text =re.sub(r"\{\|.*?\|\}", "", text, flags=re.DOTALL)

    # Remove files and figures
    text = re.sub(r"\[\[File:.*?\]\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[\[Bestand:.*?\]\]", "", text, flags=re.IGNORECASE)

    # Process links [[...|visible text]] and removes [[...]]
    text = re.sub(r"\[\[.*?\|(.*?)\]\]", r"\1", text)
    text = re.sub(r"\[\[.*?\]\]", "", text)

    # Remove URLs
    text = re.sub(r"http[s]?://\S+", "", text)

    # Remove section headers and HTML-tags
    text = re.sub(r"==.*?==", "", text)
    text = re.sub(r"<.*?>", "", text)

    # Remove unnecessary characters
    text =  re.sub(r"[{}|]", "", text)

    # Keep only the text after the first title
    match = re.search(r"'''|==", text)
    if match:
        text = text[match.start():]

    # Remove extra whitespace and newlines
    text =re.sub(r"\s+", " ", text).strip()
    
    # Split sentences and filter meaningful content
    sentences = re.split(r"\.\s+", text)
    sentences = [s.strip() for s in sentences if len(s.split()) > 3]

    # Concatenate clean text
    cleaned_text = ". ".join(sentences).strip() + "."

    return cleaned_text


def parse_xml(input_file):
    """ """
    tree = ET.parse(input_file)
    root = tree.getroot()
    
    # get namespace
    namespace = {"ns": "http://www.mediawiki.org/xml/export-0.11/"}

    # searches for <page> elements with namespace
    return root.findall("ns:page", namespace) 
