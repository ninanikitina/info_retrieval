from nltk.tokenize import RegexpTokenizer
import pandas as pd
import csv
import time
from nltk.stem import WordNetLemmatizer
from nltk.corpus.reader.wordnet import VERB, NOUN, ADJ, ADV
from nltk.tag import StanfordPOSTagger
from nltk.corpus import stopwords
from tqdm import tqdm
from functools import lru_cache
from collections import defaultdict
import matplotlib.pyplot as plt
import pickle
from collections import Counter
import math


class IndexTerm(object):
    def __init__(self, tokenized_df):
        self.term_freq = self._build_index(tokenized_df)
        self.doc_max_term = self._build_doc_max(tokenized_df)
        self.docs_num = len(tokenized_df.index)

    def _build_index(self, tokenized_df):
        content = tokenized_df["all_content"].to_list()
        id = tokenized_df["id"].to_list()
        index_terms = {}
        print("********************************")
        print("Build index")
        for i, terms in enumerate(tqdm(content)):
            for term in terms:
                if term in index_terms:
                    index_terms[term][id[i]] = index_terms[term].get(id[i], 0) + 1
                else:
                    index_terms[term] = {id[i]: 1}
        # need to save so we can use it later
        return index_terms

    def _build_doc_max(self, tokenized_df):
        '''
         number of times the most frequently-occurred term appears in d
        :param tokenized_df:
        :return: doc_max_term
        '''
        content = tokenized_df["all_content"].to_list()
        id = tokenized_df["id"].to_list()
        doc_max_term = {}
        print("********************************")
        print("Build max words")
        for i, terms in enumerate(tqdm(content)):
            c = Counter(terms)
            max_value = max(c.values()) if len(c) is not 0 else 0
            doc_max_term[id[i]] = max_value
        return doc_max_term

    def rank_docs(self, query, max_num):
        idf_words = []
        for w in query:
            if w in self.term_freq:
                word_doc_num = len(self.term_freq[w])
                idf_w = math.log2(self.docs_num / word_doc_num)
                idf_words.append(idf_w)
            else:
                idf_words.append(0)
        doc_scores = {}
        all_docs = set()
        for i, w in enumerate(query):
            if w in self.term_freq:
                docs = set(k for k, v in self.term_freq[w].items())
                all_docs.update(docs)

        for d in all_docs:
            max_d = self.doc_max_term.get(d)
            relevance_score = 0
            for i, w in enumerate(query):
                if idf_words[i] == 0:
                    continue
                frequency_w_d = self.term_freq.get(w).get(d)
                tf_w_d = frequency_w_d / max_d if frequency_w_d is not None else 0
                relevance_score += tf_w_d * idf_words[i]
            doc_scores[d] = relevance_score
        element_num = max_num if max_num < len(doc_scores) else len(doc_scores)

        return sorted(doc_scores, key=doc_scores.get, reverse=True)[:element_num]
