
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import re
from nltk.stem import WordNetLemmatizer
import pickle
import math

class GenerateSnippets:
    def __init__(self):

        #corpus_df_name = "C:\\Users\\steph\\source\\repos\\info_retrieval\\Project1\\preprocessed_files\\dataframe.ftr"
        #tokenized_df_name = "C:\\Users\\steph\\source\\repos\\info_retrieval\\Project1\\preprocessed_files\\tokenized_dataframe.ftr"
        self.lemmatizer = WordNetLemmatizer()
        
#        self.a = pd.read_feather(tokenized_df_name, columns=None, use_threads=True)
#        self.b = pd.read_feather(corpus_df_name, columns=None, use_threads=True)

        #termsList = self.a['all_content'][0]
        #f = open("C:\\Users\\steph\\source\\repos\\info_retrieval\\Project1\\myPickle", "wb")
        #pickle.dump(termsList, f)
        #f.close()

        f = open("C:\\Users\\steph\\source\\repos\\info_retrieval\\Project1\\myPickle", "rb")
        termsList = pickle.load(f)
        f.close()
        f = open("C:\\Users\\steph\\source\\repos\\info_retrieval\\Project1\\docPickle", "rb")
        doc = pickle.load(f)
        f.close()
 
        query = "Morocco is a place with relations"

        title, one, two = self.getSnippets(query, doc, termsList)
        print(title)
        print(one)
        print(two)

    
    def getSnippets(self, query, document, termsList):
        """ Gets top two top-ranked sentences of document related to query
        :based on cosine similarity iwth respect to q
        :uses TF-IDF for term weighting scheme
        :returns Title, top-ranked sentence, sencond top-ranked sentence
        """
        sentDict = self.documentTFIDF(document, termsList)
        queryDict = self.queryTFIDF(query, termsList)
            
        self.sentenceList = []
        self.resultList = []

        for s in sentDict:
            x = self.cosine(queryDict[query], sentDict[s], termsList)
            found = False
            for inx,r in enumerate(self.resultList):
                if x > r:
                    self.resultList.insert(inx, x)
                    self.sentenceList.insert(inx, s)
                    found = True
                    break
            if not found:
                self.resultList.append(x)
                self.sentenceList.append(s)
        return self.getTitle(), self.sentenceList[0], self.sentenceList[1]


    def getFullSortedSentenceList(self):
        return self.sentenceList
    
    def getFullSortedResultList(self):
        return self.resultList

    def getTitle(self):
        return self.title
 
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
        value = sum(prod) / (math.sqrt(sum(magQ)) * math.sqrt(sum(magS)))
        return value

       
    def queryTFIDF(self, query, termsList):
        """ Calculates TF-IDF for each term in query
        :seperates the title from the contents    
        :takes the document contents and splits them into sentences
        :returns dicionary with query and dictionary of terms and TFIDF --> {'query': {'term': TFIDF}}
        :there is only one query with multiple terms
        """
        sentencesList = [query]
        return self.calculateTFIDF(sentencesList, termsList)



    def documentTFIDF(self, docContent, termsList):
        """ Calculates TF-IDF for each term in each sentence
        :seperates the title from the contents    
        :takes the document contents and splits them into sentences
        :returns dicionary of each sentence with dictionary of terms and TFIDF --> {'sentence': {'term': TFIDF}}
        """
        dummies = docContent.split('\r\n\r\n')
        self.title = dummies[0]
        body = "".join(dummies[1:])
        sentencesList = re.split('[.!?]', body)
        sentencesList = [x for x in sentencesList if len(x) > 0]
        return self.calculateTFIDF(sentencesList, termsList)


    def calculateTFIDF(self, sList, termsList):
        """ Calculates TF-IDF for each sentence in sList
        :TF-IDF = (frequency(t,r) / total tokens in r) * (# of resources in the collection / # of resources in which t appears) 
        :for each sentence in sList, TF-IDF is calculated for each term in termsList
        :returns dicionary of each sentence with dictionary of terms and TFIDF --> {'sentence': {'term': TFIDF}}
        """
        docSent = {}
        numCollections = len(sList)
        DF = self.calculateDF(termsList)
        for s in sList:
            TF_IDF = {}
            TF = self.calculateTF(s, termsList)
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
        #temp= []
        #for w in processed:
        #    temp.append(self.lemmatizer.lemmatize(w))
        processedList = []
        for t in processed:
            if t in termsList:
                processedList.append(t)
        TF = self.calculateDF(processedList)
        numTerms = len(processedList)
        for term in processedList:
            TF[term] = round((TF[term] / numTerms), 4)
        return TF
        

if __name__ == "__main__":
    gs = GenerateSnippets()





