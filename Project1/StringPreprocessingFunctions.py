from nltk.corpus import stopwords
import string
from nltk.tokenize import word_tokenize as Tokenizer

def preprocess_string(document):
    document = [word for word in document if not word in stopwords.words()]
    # Remove Puntuation
    for word in document:
        for character in word:
                if character in string.punctuation:
                    word = word.replace(character, "")

    # Lemmatize Words
    return document



