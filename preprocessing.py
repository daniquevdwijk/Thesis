#
# File name: preprocessing.py
# Author: Danique van der Wijk
# Student number: s3989771
# Last updated: 3 February 2025
# Description: The file where the preprocessing of the Wikipedia dataset takes place
#

import re
import xml.etree.ElementTree as ET
import html

def extract_text(page):
    """
    Extracts and cleans the text content from a given XML page element.
    Args:
        page (Element): An XML element representing a page, which contains
                        <revision> and <text> tags in the MediaWiki XML format.
    Returns:
        str: The cleaned text content extracted from the <text> tag within the
             <revision> tag. Returns an empty string if the <revision> or <text>
             tags are not found, or if the <text> tag is empty.
    """
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
    cleaned_text = clean_wikitext(raw_text).strip()

    return cleaned_text


def clean_wikitext(text):
    """
    Cleans and processes wikitext by removing or transforming various elements.
    Args:
        text (str): The wikitext to be cleaned.
    Returns:
        str: The cleaned and processed text.
    The function performs the following operations:
    - Removes templates enclosed in {{...}}.
    - Keeps the content of <ref>...</ref> tags but removes the tags themselves.
    - Removes self-closing <ref.../> tags.
    - Removes tables enclosed in {|...|}.
    - Removes files and figures specified with [[File:...]] or [[Bestand:...]].
    - Processes links of the form [[...|visible text]] to keep only the visible text.
    - Removes links of the form [[...]].
    - Removes URLs.
    - Removes section headers and HTML tags.
    - Removes unnecessary characters such as {, }, and |.
    - Keeps only the text after the first title (indicated by ''' or ==).
    - Removes extra whitespace and newlines.
    - Splits the text into sentences and filters out short sentences.
    - Concatenates the cleaned sentences into a single string.
    - Unescapes HTML entities and removes non-breaking spaces.
    """
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
    cleaned_text = html.unescape(cleaned_text).replace('\xa0', ' ').strip()
    cleaned_text = " ".join(cleaned_text.split())


    return cleaned_text


def parse_xml(input_file):
    """
    Parses an XML file and returns a list of <page> elements.
    Args:
        input_file (str): The path to the XML file to be parsed.
    Returns:
        list: A list of <page> elements found in the XML file.
    """
    tree = ET.parse(input_file)
    root = tree.getroot()
    
    # get namespace
    namespace = {"ns": "http://www.mediawiki.org/xml/export-0.11/"}

    # searches for <page> elements with namespace
    return root.findall("ns:page", namespace) 
