#
# File name: evaluation.py
# Author: Danique van der Wijk
# Student number: s3989771
# Last updated: 8 January 2025
# Description: The file with the code to execute the Support Vector Machine (SVM) and the calculation of the BLEU scores
#


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from nltk.translate.bleu_score import sentence_bleu
from transformers import BertTokenizer, BertForSequenceClassification, pipeline, AutoTokenizer, AutoModel, AutoModelForCausalLM
from bert_score import score
import language_tool_python
import Levenshtein
import pandas as pd
import numpy as np
import torch
import spacy
import re

nlp = spacy.load("nl_core_news_sm")

def label_text(stego_text, cover_text):
    """ """
    # Load model
    model = "wietsedv/bert-base-dutch-cased"
    tokenizer = BertTokenizer.from_pretrained(model)
    bert_model = BertForSequenceClassification.from_pretrained(model, num_labels=2)

    # Define a classificiation pipeline
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)

    # Combine the stego and cover text
    X = cover_text + stego_text
    y = []

    for text in X:
        # use the model to classify the text as natural or unnatural
        scores = classifier(text)
        # 0 corresponds with natural and 1 with unnatural
        label = int(np.argmax([s['score'] for s in scores[0]]))
        y.append(label)

    return X, y

def train_svm(stego_text, cover_text):
    """ """
    # Label the text
    X, y = label_text(stego_text, cover_text)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)

    # Convert to numerical using TF-IDF
    vectorizer = TfidfVectorizer()
    x_train_trans = vectorizer.fit_transform(X_train)
    x_test_trans = vectorizer.transform(X_test)

    # Train the SVM
    model = SVC(kernel='linear', random_state=23)
    model.fit(x_train_trans, y_train)

    # Evaluate te model
    y_pred = model.predict(x_test_trans)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy


def calculate_bleu_score(stego_text, cover_text, weights):
    """ """
    # Split the input texts into sentences
    stego_sentences = stego_text.split(".")
    cover_sentences = cover_text.split(".")

    # Check if the number of sentences is the same
    if len(stego_sentences) != len(cover_sentences):
        raise ValueError("The number of sentences in the stego text and cover text must be the same.")

    bleu_scores = []

    # Calculate BLEU score for each sentence
    for cover_sentence, stego_sentence in zip(cover_sentences, stego_sentences):
        reference = cover_sentence.split()
        candidate = stego_sentence.split()
        score = sentence_bleu([reference], candidate, weights=weights)
        bleu_scores.append(score)
    
    # Calculate average BLEU score
    avg_bleu_score = sum(bleu_scores) / len(bleu_scores)
    #print(f"Average BLEU score: {round(avg_bleu_score, 3)}")

    return avg_bleu_score


def check_grammar(cover_text, stego_text):
    """ java nodig? ja """
    # Make a LanguageTool object for Dutch
    tool = language_tool_python.LanguageTool('nl')

    # Check cover_text and stego_text
    cover_corrections = tool.check(cover_text)
    stego_corrections = tool.check(stego_text)

    # Show amount of detected grammar errors
    print(f"Amount of grammatical issues in cover_text: {len(cover_corrections)}")
    print(f"Amount of grammatical issues in stego_text: {len(stego_corrections)}")

    return cover_corrections, stego_corrections

def count_syllables(word: str) -> int:
    """ """
    vowels = "aeiouAEIOUáéíóúÁÉÍÓÚäëïöüÄËÏÖÜ"

    # Find all groups of contiguous vowels
    groups = re.findall(r'[{}]+'.format(vowels), word)

    # Every word has at least one syllable
    return max(1, len(groups))

def calculate_flesch_douma(text: str) -> float:
    """ """
    # Parse the text with spaCy
    doc = nlp(text)

    # Count sentences, words and syllables
    total_sentences = 0
    total_words = 0
    total_syllables = 0

    for sent in doc.sents:
        total_sentences += 1
        for token in sent:
            # skip punctuation
            if token.is_alpha:
                total_words += 1
                total_syllables += count_syllables(token.text)

    # If there are no words or sentences, give 0.0 to avoid error
    if total_sentences == 0 or total_words == 0:
        return 0.0
    
    words_sentence = total_words / total_sentences
    syllables_word = total_syllables / total_words

    # Flesch-Douma formula
    flesch_douma = (206.835 - (0.93 * words_sentence) - (77.0 * syllables_word))

    return flesch_douma

def calc_bertscore(cover_text, stego_text):
    """ """
    model_name = "pdelobelle/robbert-v2-dutch-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    cover_doc = nlp(cover_text)
    stego_doc = nlp(stego_text)

    cover_sents = [sent.text.strip() for sent in cover_doc.sents]
    stego_sents = [sent.text.strip() for sent in stego_doc.sents]

    # Make sure the number of sentences is the same
    min_len = min(len(cover_sents), len(stego_sents))
    cover_sents = cover_sents[:min_len]
    stego_sents = stego_sents[:min_len]

    # Calculate BERTScore
    p, r, f1 = score(stego_sents, cover_sents, lang='nl')

    # Return the results as a dictionary
    return {"precision": p.mean().item(), "recall": r.mean().item(), "f1": f1.mean().item()}

def perplexity(text: str, model_name: str = "gpt2", stride: int = 512) -> float:
    """ """
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    # Encode text
    encodings = tokenizer(text, return_tensors='pt')

    max_length = model.config.n_positions
    input_ids = encodings.input_ids
    seq_len = input_ids.size(1)

    nlls = []
    # process the text in parts of the size 'stride'
    for i in range(0, seq_len, stride):
        begin_loc = i
        end_loc = min(i + stride, seq_len)

        # For the first segment, the full length is taken as target
        # For the other segments we only count the new tokens
        trg_len = end_loc - (begin_loc if i > 0 else 0)

        input_ids_slice = input_ids[:, begin_loc:end_loc]
        target_ids = input_ids_slice.clone()

        # Make sure only the tokens we added in this batch are counted in the loss
        if i > 0:
            target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids_slice, labels = target_ids)
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)
    
    # Perplexity = exp(NLL / token count)
    ppl = torch.exp(torch.stack(nlls).sum() / seq_len)
    return ppl.item()

def levenshtein_dist(word1, word2):
    """ """
    distance = Levenshtein.distance(word1, word2)
    return distance
