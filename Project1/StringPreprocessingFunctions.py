import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from nltk.tokenize import word_tokenize as Tokenizer

def preprocess_string(document):

    # Remove Puntuation
    for character in document:
                if character in string.punctuation:
                    document = document.replace(character, "")

    # Tokenizes text and then removes stopwords
    text_tokens = Tokenizer(document)
    tokens_without_stopwords = [word for word in text_tokens if not word in stopwords.words()]

    # Lemmatize Words
    lemmed_words = []
    lemmatizer = WordNetLemmatizer()
    words = document.split()
    for word in words:
        lemmed_words.append(lemmatizer.lemmatize(word))

    return words



