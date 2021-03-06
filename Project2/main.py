import time
import pickle
import pandas as pd
import json
from Project2.preprocessing import tokanize_text
from Project2.preprocessing import tokenize_query
from Project1.GenerateSnippets_v2 import GenerateSnippets
from Project2.QueryLogger import QueryLogger



class SearchEngineData():
    def __init__(self):
        self.a = 1
        self.query_logger = QueryLogger()

    def  run_query(self, query_sent="No query submitted", add_entities=False):
        if query_sent == "No query submitted":
            return "No query submitted"
        print("Uploading data...")

        #myPath = "C:\\Users\\steph\\source\\repos\\info_retrieval\\Project2\\"
        myPath = ""
        corpus_df_name = myPath + "preprocessed_files\\article_df.ftr"
        index_file_name = myPath + "preprocessed_files\\articles_index.pickle"
        results_file_name = myPath + "preprocessed_files\\rank.csv"

        with open(index_file_name, 'rb') as handle:
            index = pickle.load(handle)
        df = pd.read_feather(corpus_df_name, columns=None, use_threads=True)
        print("Data successfully uploaded")
        df.to_csv('test.csv')

        print("Building rank...")
        if add_entities:
            test_query_w_entities = tokenize_query(query_sent, lower=True, remove_digits=True, is_lemmatized=True,
                                                  remove_stop_words=True)
            rank, all_rank = self.index.rank_docs_w_entities(test_query_w_entities, 5, entities_coeff=5)
        else:
            query = tokanize_text(query_sent, lower=True, remove_digits=True, is_lemmatized=True,
                                  remove_stop_words=True)
            rank, all_rank = index.rank_docs(query, max_num=5)

        print(all_rank)
        print(rank)
        print("Rank is created")
       
        print("Generating snippets...")
        snippet_obj = GenerateSnippets(index.get_terms(), df)
        snippets_df = snippet_obj.getSnippets(query_sent, rank)
        # snippets_df.to_csv(results_file_name)
        print("Snippets generated")

        snippets_json = json.dumps(snippets_df, indent = 4)

        print(snippets_json)

        return snippets_json




def main():
    tester = SearchEngineData()
    tester.run_query(query_sent="To what position was Malenkov demoted?", add_entities=True)
    test_query_tokanized = tokenize_query
    a = 1


if __name__ == "__main__":
    main()

