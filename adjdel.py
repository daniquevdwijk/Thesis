#
# File name: adjdel.py
# Author: Danique van der Wijk
# Student number: s3989771
# Last updated: 23 December 2024
# Description: The file where the adjective deletion takes place
#

import spacy


# Loads the Dutch language model of spaCy
nlp = spacy.load("nl_core_news_sm")


def bitcode_to_text(text, bitcode):
    """ """
    doc = nlp(text)
    sentences = list(doc.sents)
    modified_sentences = []

    for i, sent in enumerate(sentences):
        # check if there are bits to encode
        if i < len(bitcode):
            bit = bitcode[i]
            if bit == "1":
                # No adjective deletion
                modified_sentences.append(sent.text)
            else:
                # Adjective deletion
                cleaned_sentence = " ".join([token.text for token in sent if token.pos_ != "ADJ"])
                modified_sentences.append(cleaned_sentence)
        else:
            # If there are no bits left, leave the rest of the text unchanged
            modified_sentences.append(sent.text)
    
    return " ".join(modified_sentences)