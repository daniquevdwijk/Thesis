# Adjective Deletion

## Needed packages

To use this code, you need to execute the following command in your terminal:

```bash
pip install nltk transformers torch spacy stanza pandas tqdm python-Levenshtein lxml
```
To download the Dutch language model for SpaCy you need to execute this:
```bash
python -m spacy download nl_core_news_sm
````
The Dutch language model for Stanza is downloaded if you run main.py.

## Run the code

To run this code, you simply need to execute the following command in your terminal:

```bash
python main.py
```

## The files

`main.py`: This file calls the other files to preprocess the data (in `preprocessing.py`), execute the adjective deletion (in `adjdel.py`) and generate the results (in `evaluation.py`).

`preprocessing.py`: In this file the preprocessing takes place. It cleans the Wikimedia file and puts the clean data in a CSV file.

`adjdel.py`: In this file the processing (adjective deletion) takes place. It tags and counts the adjectives and removes or keeps them in a text.

`evaluation.py`: In this file the results are generated. The results are: Levenshtein distance, perplexity scores and the BLEU scores.


