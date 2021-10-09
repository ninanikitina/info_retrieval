import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize as Tokenizer
import pandas as pd
from nltk.corpus import stopwords
import numpy as np

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
punctuation = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

###############################
# Uncomment and download once #
# nltk.download()             #
###############################

# Our index
index = {}

# Document List
docs = [
    "Old-school musical numbers, feisty princesses, funny sidekicks and a mix of action, comedy and romance come together in Frozen, a Disney animation that works hard to keep everyone happy.",
    "Anyone with an appreciation for old-school musical numbers will find something pleasantly familiar and cozy about Frozen.",
    "Frozen, the new animated film by Disney earns its charms the honest way: with smart writing and heartfelt performances.",
    "The animation is simply superb. Ice has never looked so good, except as the real thing. Technical precision and innovation are expected nowadays in computer animation but Frozen combines that with a gorgeously rich design.",
    ": An enjoyable fairy tale romp with some clever plot twists thrown in and positive messages for young girls that subvert those usually provided by traditional Disney princesses."
 ]

for x in range (0, len(docs)):

    # Lowers text
    docs[x] = docs[x].lower()

    # Remove Punctuation Via Loop
    for character in docs[x]:
        if character in punctuation:
            docs[x] = docs[x].replace(character, "")

    # Tokenizes text and then removes stopwords
    text_tokens = Tokenizer(docs[x])
    tokens_without_stopwords = [word for word in text_tokens if not word in stopwords.words()]
    
    # Stems and Lemmatizes Word and Adds to Index
    lemmatized_words = []
    stemmed_words = []
    for word in tokens_without_stopwords:
        stemmed_word = stemmer.stem(word) # Creates stemmed word
        stemmed_words.append(stemmed_word)
        # print("Stemmed: ", stemmed_word)
        lemmatized_word = lemmatizer.lemmatize(word, pos='a') # Creates lemmatized word
        lemmatized_words.append(lemmatized_word)
        # print("Lemmatized: ", lemmatized_word)
        # print()

        # if stemmed_word not in index:
        #     index[stemmed_word] = set() # Creates a set if nothing found
        # if stemmed_word in index:
        #     index[stemmed_word].add(x+1) # Adds doc num if found

        if lemmatized_word not in index:
            index[lemmatized_word] = set() # Creates a set if nothing found
        if lemmatized_word in index:
            index[lemmatized_word].add(x+1) # Adds doc num if found

    # Prints number of words in each document to calculate TF-IDF
    print("Doc" + str(x + 1) + " lemmatized word num: " + str(len(lemmatized_words)))
    print(lemmatized_words)
    print("Doc" + str(x + 1) + " stemmed word num: " + str(len(stemmed_words)))

# Print index
print("----- INDEX -----")
print("___________________________")
for word in sorted(index):
    print(f"{word}: {index[word]}")
print("___________________________")


# Create a term-document matrix
term_doc_matrix = np.zeros((len(index), len(docs)), dtype=np.uint8)
for idx, word in enumerate(sorted(index)):
    term_doc_matrix[idx, [doc_id - 1 for doc_id in index[word]]] = 1
df_term_doc = pd.DataFrame(term_doc_matrix, columns=["doc1", "doc2", "doc3", "doc4","doc5"], index=sorted(index))

# Create a term-term matrix
term_term_matrix = np.matmul(term_doc_matrix, term_doc_matrix.transpose())
df_term_term = pd.DataFrame(term_term_matrix, columns=sorted(index), index=sorted(index))

# Find two tems with specified correlation,
# 0 for laest correlated
# from 2 to 4 for most correlated for this specific collection
correlation_coef = 4
for i in range(len(index) -1):
    for j in range(len(index) -1):
        if df_term_term.iloc[i, j] == correlation_coef:
            print(df_term_term.columns[j] + " + " + df_term_term.index[i])




#def get_weight(frequency, total_tokens, total_resources, token_resources):
    
#    weight = (frequency/total_toks) * (total_resources/tok_resources)
#    print(weight)
#    return weight

#if __name__ == "__main__":
#    main()

#class Index:
#    def __init__(self):
#        self.index = {}
#        self.total_documents = 0
#        self.total_words = 0

#    def add_document(document):
#            document = document.lower()
#            self.total_documents += 1
#        # Remove Punctuation Via Loop
#            for character in document:
#                if character in punctuation:
#                    document = document.replace(character, "")

#            # Tokenizes text and then removes stopwords
#            text_tokens = Tokenizer(document)
#            tokens_without_stopwords = [word for word in text_tokens if not word in stopwords.words()]
    
#            # Stems and Lemmatizes Word and Adds to Index
#            for word in tokens_without_stopwords:
#                if lemmatized_word not in index:
#                    index[lemmatized_word] = set() # Creates a set if nothing found
#                    self.total_words += 1
                    
#                if lemmatized_word in index:
#                    index[lemmatized_word].add(x+1) # Adds doc num if found
#   def print(self):
#       for word in self.index:
#       print(f"{word}: {self.index[word]}")

                    