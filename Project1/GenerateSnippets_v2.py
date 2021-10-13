
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
import pickle
import math

class GenerateSnippets:
    def __init__(self, termsList, df):
        self.termsList = termsList
        self.df = df
        self.lemmatizer = WordNetLemmatizer()


    def getTermsForId(self,id):
        localTerms = []
        for term in self.termsList.keys():
            ids_in_term = self.termsList[term].keys()
            if id in ids_in_term and term not in localTerms:
                localTerms.append(term)
        print("Local terms: %s" % localTerms)
        return localTerms

    
    def getSnippets(self, query, document_ids):
        """ Gets top two top-ranked sentences of document related to query
        :based on cosine similarity iwth respect to q
        :uses TF-IDF for term weighting scheme
        :returns Title, top-ranked sentence, sencond top-ranked sentence
        """

        snippets_df = pd.DataFrame(columns=["title", "one", "two"])

        for id in document_ids:
            selected_row = self.df[self.df["id"] == id]
            title = selected_row["title"].to_list()[0]
            document = selected_row["content"].to_list()[0]
            print("Contents: %s" % document)
            #localTerms = self.getTermsForId(id)
            sentDict = self.documentTFIDF(document)
            queryDict = self.queryTFIDF(query)
            sentenceList = []
            resultList = []

            for s in sentDict:
                x = self.cosine(queryDict[query], sentDict[s], self.termsList)
                #x = self.cosine(queryDict[query], sentDict[s], localTerms)
                found = False
                for inx,r in enumerate(resultList):
                    if x > r:
                        resultList.insert(inx, x)
                        sentenceList.insert(inx, s)
                        found = True
                        break
                if not found:
                    resultList.append(x)
                    sentenceList.append(s)
            dict = {"title": title, "one": sentenceList[0], "two": sentenceList[1]}
            print(dict)
            snippets_df = snippets_df.append(dict, ignore_index=True)

        return snippets_df


    # def getFullSortedSentenceList(self):
    #     return self.sentenceList
    #
    # def getFullSortedResultList(self):
    #     return self.resultList
    #
    # def getTitle(self):
    #     return self.title
    #
    def calculateDF(self, terms):
        """ Calculates DF for a list of terms
            :Takes in a list of terms 
            :Calculates how many times a term is in the list
            :Returns dictionary of terms with with DF --> {'term': DF}
        """
        DF = {}
        for i in terms:
            DF.setdefault(i,0)
            DF[i] += 1
        return DF

    def cosine(self, query, sentence, termsList): 
        """ Gets cosine value
            :Passes terms to have a fixed order for vector space components
            :query is a dictionary where key is the term and value is TFIDF --> {'term': TFIDF}
            :sentence is a dictionary where key is the term and value is TFIDF --> {'term': TFIDF}
            :termList is a list of terms
            :Returns cosine value of sentence and query
        """
        prod = []
        magQ = []
        magS = []

        for t in termsList:
            q = query[t]
            s = sentence[t] 
            prod.append(q * s)
            magQ.append(q**2)
            magS.append(s**2)
        if (math.sqrt(sum(magQ)) * math.sqrt(sum(magS))) != 0:
            value = sum(prod) / (math.sqrt(sum(magQ)) * math.sqrt(sum(magS)))
        else:
            value = 0
        return value

       
    def queryTFIDF(self, query):
        """ Calculates TF-IDF for each term in query
        :seperates the title from the contents    
        :takes the document contents and splits them into sentences
        :returns dicionary with query and dictionary of terms and TFIDF --> {'query': {'term': TFIDF}}
        :there is only one query with multiple terms
        """
        sentencesList = [query]
        return self.calculateTFIDF(sentencesList)



    def documentTFIDF(self, body):
        """ Calculates TF-IDF for each term in each sentence
        :seperates the title from the contents    
        :takes the document contents and splits them into sentences
        :returns dicionary of each sentence with dictionary of terms and TFIDF --> {'sentence': {'term': TFIDF}}
        """
        dummies = body.split('\r\n\r\n')
        body = " ".join(dummies)
        sentencesList = sent_tokenize(body)
        sentencesList = [x for x in sentencesList if len(x) > 0]
        return self.calculateTFIDF(sentencesList)


    def calculateTFIDF(self, sList):
        """ Calculates TF-IDF for each sentence in sList
        :TF-IDF = (frequency(t,r) / total tokens in r) * (# of resources in the collection / # of resources in which t appears) 
        :for each sentence in sList, TF-IDF is calculated for each term in termsList
        :returns dicionary of each sentence with dictionary of terms and TFIDF --> {'sentence': {'term': TFIDF}}
        """
        docSent = {}
        numCollections = len(sList)
        DF = self.calculateDF(self.termsList)
        for s in sList:
            TF_IDF = {}
            TF = self.calculateTF(s, self.termsList)
            for key in DF:
                TF_IDF.setdefault(key,0)
                TF_IDF[key] = round(TF.get(key, 0) * (numCollections / DF[key]), 4)
            docSent[s] = TF_IDF 
        return docSent


    def removePunctuation(self, sentence):
        """ Removes punctuation from sentence
        :Returns sentence without punctuationS
        """
        punc = ['.', '!', ',', '?', ':', ';']
        for i in punc:
            sentence = sentence.replace(i , "")
        return sentence
            
    def calculateTF(self, sentence, termsList):
        """ Calculates TF each term in a sentence
        :TF = frequency(t,r) / total tokens in r
        :lowercases all chars, removes punctuation and digits from sentence
        :gets DF of all terms in sentence that are in termsList
        :calculates TF of each term with DF and num terms in sentence
        :Returns TF dictionary with each term in sentence and TF rounded --> {'term': TF}
        """
        sentence = sentence.lower()
        sentence = self.removePunctuation(sentence)
        processed = sentence.split()
        processed = [word for word in processed if not word.isdigit()]
        temp = []
        for w in processed:
           temp.append(self.lemmatizer.lemmatize(w))
        processedList = []
        for t in temp:
            if t in termsList:
                processedList.append(t)
        TF = self.calculateDF(processedList)
        numTerms = len(processedList)
        for term in processedList:
            TF[term] = round((TF[term] / numTerms), 4)
        return TF
        

if __name__ == "__main__":
    myp = "C:\\Users\\steph\\source\\repos\\info_retrieval\\Project1\\"
    index_file_name = myp +"preprocessed_files\\wiki_index.pickle"  # TODO: Download file from google drive and move to the preprocessed_files folder in the Project1 directory
    corpus_df_name = myp +"preprocessed_files\\wiki_df.ftr"  

    with open(index_file_name, 'rb') as handle:
        index = pickle.load(handle)
    df = pd.read_feather(corpus_df_name, columns=None, use_threads=True)
    #gs = GenerateSnippets(index.term_freq, df)
    gs = GenerateSnippets(index.get_terms(), df)
    gs.getSnippets("shiite",[1])





