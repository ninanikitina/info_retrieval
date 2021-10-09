import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize as Tokenizer

class Index:
    def __init__(self):
        self.index = {}
        self.total_documents = 0
        self.total_words = 0
        self.lemmatizer = WordNetLemmatizer()

    def add_document(self, document, document_name = "unknown"):
            document = document.lower()
            self.total_documents += 1
            if document_name == "unknown":
                document_name == f"Unknown Document: #{self.total_documents}"
            # Remove Punctuation Via Loop
            for character in document:
                if character in string.punctuation:
                    document = document.replace(character, "")

            # Tokenizes text and then removes stopwords
            text_tokens = Tokenizer(document)
            tokens_without_stopwords = [word for word in text_tokens if not word in stopwords.words()]
    
            #Lemmatizes Word and Adds to Index
            for word in tokens_without_stopwords:
                lemmatized_word = self.lemmatizer.lemmatize(word) 
                if lemmatized_word not in self.index:
                    self.index[lemmatized_word] = {} # Creates a nested dictionary
                    self.total_words += 1

                if lemmatized_word in self.index:
                    if document_name not in self.index.get(lemmatized_word):
                        self.index[lemmatized_word][document_name] = 1
                    else:
                        frequency = self.index[lemmatized_word][document_name]
                        self.index[lemmatized_word][document_name] = frequency+1
                   
                
                    
                  # Adds doc num if found

   
    def print(self):
        for word in self.index:
            print(f"{word}: {self.index[word]}")
            print("Word contains this many documents: ", len(self.index[word]))

if __name__ == "__main__":
    index = Index()
    index.add_document("THIS IS A SILLY TEST OF MY INDEX! INDEX INDEX INDEX", document_name = "TEST INDEX")
    index.add_document("ANOTHER TEST INDEX DASDFK", document_name = "ANOTHER INDEX")
    
    index.print()