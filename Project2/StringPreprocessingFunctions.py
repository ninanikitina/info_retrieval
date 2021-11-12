from nltk.corpus import stopwords
import string
import spacy
from nltk.tokenize import word_tokenize as Tokenizer


def preprocess_string(document):

    stop_words = set(stopwords.words('english'))
    nlp = spacy.load("en_core_web_sm")
    doc_nlp = nlp(document)
    nouns = [chunk.text.lower() for chunk in doc_nlp.noun_chunks]
    nouns = [noun for noun in nouns if noun not in stop_words]

    return {
        "original_query": document, 
        "nouns": nouns
        }


