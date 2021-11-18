import time
import pickle
import pandas as pd
import json
from tqdm import tqdm
from Project2.preprocessing import tokanize_text
from Project1.GenerateSnippets_v2 import GenerateSnippets
from Project2.preprocessing import tokenize_query
import nltk
nltk.download('averaged_perceptron_tagger')



class SearchEngineData():
    def __init__(self):
        myPath = ""
        self.corpus_df_name = myPath + "preprocessed_files\\article_df.ftr"
        self.index_file_name = myPath + "preprocessed_files\\articles_index.pickle"
        self.results_file_name = myPath + "preprocessed_files\\rank.csv"

        with open(self.index_file_name, 'rb') as handle:
            self.index = pickle.load(handle)
        self.df = pd.read_feather(self.corpus_df_name, columns=None, use_threads=True)
        print("Index was successfully uploaded")

    def run_query(self, query_sent="No query submitted"):
        query = tokanize_text(query_sent, lower=True, remove_digits=True, is_lemmatized=True, remove_stop_words=True)
        rank, all_rank = self.index.rank_docs(query, max_num=5)

        snippet_obj = GenerateSnippets(self.index.get_terms(), self.df)
        snippets_df = snippet_obj.getSnippets(query_sent, rank)
        return snippets_df

    def run_query_entities(self, query_sent, entities_coeff):
        test_query_tokanized = tokenize_query(query_sent, lower=True, remove_digits=True, is_lemmatized=True, remove_stop_words=True)
        rank, all_rank = self.index.rank_docs_w_entities(test_query_tokanized, 5, entities_coeff)
        snippet_obj = GenerateSnippets(self.index.get_terms(), self.df)
        snippets_df = snippet_obj.getSnippets(query_sent, rank)
        return snippets_df



def main():
    # Normalized Discounted Cumulative Gain will be calculated based found data
    ndcg = pd.DataFrame(columns=["question", "answer", "doc num", "sentences num", "answer sentence"])
    myPath = ""
    questions_df_name = myPath + "preprocessed_files\\questions_df.ftr"
    ndcg_file_name = myPath + "preprocessed_files\\ndcg_df.ftr"
    tester = SearchEngineData()
    questions_df = pd.read_feather(questions_df_name, columns=None, use_threads=True)
    questions = questions_df["question"].values.tolist()
    answers = questions_df["answer"].values.tolist()
    questions[0] = "What century did the Normans first gain their separate identity?"
    answers[0] = "10th century"
    for i, question in enumerate(tqdm(questions)):
        # snippets = tester.run_query(question)
        snippets = tester.run_query_entities(question, entities_coeff=5)
        answer = answers[i]
        doc_num = 0
        sentence_num = 0
        answer_sent = "n/a"
        for key, value in snippets.items():
            if answer.lower() in value['one'].lower():
                doc_num = key + 1
                sentence_num = 1
                answer_sent = value['one']
            if answer.lower() in value['two'].lower():
                doc_num = key + 1
                sentence_num = 2
                answer_sent = value['two']
        ndcg = ndcg.append({"question": question, "answer": answer,
                            "doc num": doc_num, "sentences num": sentence_num,
                            "answer sentence": answer_sent},
                           ignore_index=True)
    ndcg.to_feather(ndcg_file_name)
    # ndcg.to_csv('out.csv')

if __name__ == "__main__":
    query_sent = "Who lay special emphasis on conservation of particular species?"
    test_query_tokanized_w_entities = tokenize_query(query_sent, lower=True, remove_digits=True, is_lemmatized=True,
                                          remove_stop_words=True)
    test_query_tokanized_basic = tokanize_text(query_sent, lower=True, remove_digits=True, is_lemmatized=True,
                                          remove_stop_words=True)
    print(test_query_tokanized_w_entities)
    print(test_query_tokanized_basic)

    main()

