from tqdm import tqdm
from collections import Counter
import math


class IndexTerm(object):
    def __init__(self, tokenized_df):
        self.term_freq = self._build_index(tokenized_df)
        self.doc_max_term = self._build_doc_max(tokenized_df)
        self.docs_num = len(tokenized_df.index)


    def get_terms(self):
        return self.term_freq.keys()


    def _build_index(self, tokenized_df):
        """
        Build and return an index_term dictionary with the following structure:
        {
        term_1: {doc_id_i: frequency, ... , doc_id_n: frequency}
        term_2: {doc_id_i: frequency ... , doc_id_n: frequency}
        ...
        term_n: {doc_id_i: frequency, ... , doc_id_n: frequency}
        }
        where doc_id_i is id of document that has specified term
        and corresponding frequency is how many times the term appears in the document
        Parameters
            ----------
            tokenized_df : df
                Data frame with 2 columns:
                "document_id" (int) | "all_content" (list of tokens)
        """
        content = tokenized_df["all_content"].to_list()
        id = tokenized_df["id"].to_list()
        index_terms = {}
        for i, terms in enumerate(tqdm(content)):
            for term in terms:
                if term in index_terms:
                    index_terms[term][id[i]] = index_terms[term].get(id[i], 0) + 1
                else:
                    index_terms[term] = {id[i]: 1}
        return index_terms


    def _build_doc_max(self, tokenized_df):
        """
        Build and return an index_term dictionary with the following structure:
        { doc_id_1: max, doc_id_2: max, ... , doc_id_n: max }

        where max  is the number of times the most frequently-occurred term appears in the document
        Parameters
            ----------
            tokenized_df : df
                Data frame with 2 columns:
                "document_id" (int) | "all_content" (list of tokens)
        """
        content = tokenized_df["all_content"].to_list()
        id = tokenized_df["id"].to_list()
        doc_max_term = {}
        for i, terms in enumerate(tqdm(content)):
            c = Counter(terms)
            max_value = max(c.values()) if len(c) != 0 else 0
            doc_max_term[id[i]] = max_value
        return doc_max_term


    def rank_docs(self, query, max_num):
        """
        Choose a set of documents CR comprised of all the documents in DC
        that contain each of the terms in q.
        If there are less than 50 documents, then consider resources in DC
        that contain n-1 terms in q (all the combinations).
        Calculate relevance score for each document and return five most rated documents.
        The 𝑅𝑒𝑙𝑒𝑣𝑎𝑛𝑐𝑒𝑆𝑐𝑜𝑟𝑒(𝑞, 𝑑) = ∑ 𝑇𝐹 (𝑤, 𝑑) × 𝐼𝐷𝐹(𝑤)
        𝑇𝐹(𝑤, 𝑑) = 𝑓𝑟𝑒𝑞(𝑤, 𝑑) / 𝑚𝑎𝑥_d
            freq(w, d) is the number of times w appears in the document
            max_d is the number of times the most frequently-occurred term appears in the document
        𝐼𝐷𝐹(𝑤) = 𝑙𝑜𝑔2(𝑁/𝑛)
            N is the number of documents in DC
            n is the number of documents in DC in which w appears at least once

            ----------
            tokenized_df : df
                Data frame with 2 columns:
                "document_id" (int) | "all_content" (list of tokens)
        """
        #  Calculate 𝐼𝐷𝐹(𝑤) = 𝑙𝑜𝑔2(𝑁/𝑛) for each term in the query
        idf_words = []
        for w in query:
            if w in self.term_freq:
                word_doc_num = len(self.term_freq[w])
                idf_w = math.log2(self.docs_num / word_doc_num)
                idf_words.append(idf_w)
            else:
                idf_words.append(0)

        #  Choose a set of documents CR comprised of all the documents in DC
        #  that contain each of the terms in q.
        all_docs_id = set()
        chosen_docs_id = set()
        docs_term_sets = []
        for i, w in enumerate(query):
            if w in self.term_freq:
                docs = set(k for k, v in self.term_freq[w].items())
                temp = docs.copy()
                docs_term_sets.append(temp)
                if i == 0:
                    chosen_docs_id = docs
                else:
                    chosen_docs_id.intersection_update(docs)
                all_docs_id.update(docs)
        #  If there are less than 50 documents, then consider resources in DC
        #  that contain n-1 terms in q (all the combinations), then n-2 ect.
        min_q = len(query) - 1
        while len(chosen_docs_id) < 50 and min_q >= 1:
            all_docs_id.difference_update(chosen_docs_id)
            for d in all_docs_id:
                count = 0
                for s in docs_term_sets:
                    if d in s:
                        count = count + 1
                if count >= min_q:
                    chosen_docs_id.add(d)
            min_q = min_q - 1

        #  𝑇𝐹(𝑤, 𝑑) = 𝑓𝑟𝑒𝑞(𝑤, 𝑑) / 𝑚𝑎𝑥_d
        doc_scores = {}
        for d in chosen_docs_id:
            max_d = self.doc_max_term.get(d)
            relevance_score = 0
            for i, w in enumerate(query):
                if idf_words[i] == 0:
                    continue
                frequency_w_d = self.term_freq.get(w).get(d)
                tf_w_d = frequency_w_d / max_d if frequency_w_d is not None else 0
                #  𝑅𝑒𝑙𝑒𝑣𝑎𝑛𝑐𝑒𝑆𝑐𝑜𝑟𝑒(𝑞, 𝑑) = ∑ 𝑇𝐹(𝑤, 𝑑) × 𝐼𝐷𝐹(𝑤)
                relevance_score = relevance_score + tf_w_d * idf_words[i]
            doc_scores[d] = relevance_score
        element_num = max_num if max_num < len(doc_scores) else len(doc_scores)

        return sorted(doc_scores, key=doc_scores.get, reverse=True)[:element_num], doc_scores
