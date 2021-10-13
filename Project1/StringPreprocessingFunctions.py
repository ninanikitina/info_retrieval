from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
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
    lemmed_words = []
    lemmatizer = WordNetLemmatizer()
    for word in document:
        lemmed_words.append(lemmatizer.lemmatize(word).lower())

    return lemmed_words



