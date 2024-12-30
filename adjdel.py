#
# File name: adjdel.py
# Author: Danique van der Wijk
# Student number: s3989771
# Last updated: 30 December 2024
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
    bit_index = 0 # keeps which bit from the bitcode is used

    for sent in sentences:
        # checks if there are adjectives in the sentence
        has_adjectives = any(token.pos_ == "ADJ" for token in sent)

        if has_adjectives:
            # only applies if there are adjectives
            if bit_index < len(bitcode):
                bit = bitcode[bit_index]
                bit_index += 1 

                if bit == "1":
                    # Keep adjectives
                    modified_sentences.append(sent.text)
                else:
                    # Delete adjectives
                    cleaned_sentence = " ".join([token.text for token in sent if token.pos_ != "ADJ"])
                    modified_sentences.append(cleaned_sentence)
            else:
                # if the bitcode is empty, add original sentence
                modified_sentences.append(sent.text)
        else:
            # add sentences without adjectives unchanged
            modified_sentences.append(sent.text)

    return " ".join(modified_sentences)