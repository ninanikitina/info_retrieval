import time
import pickle
import pandas as pd
import json
from Project1.preprocessing import tokanize_text
from Project1.GenerateSnippets_v2 import GenerateSnippets



class SearchEngineData():
    def __init__(self):
        self.a = 1

    def run_query(self, query_sent="No query submitted"):
        if query_sent == "No query submitted":
            return "No query submitted"
        print("Uploading data...")

        myPath = ""
        corpus_df_name = myPath + "preprocessed_files\\article_df.ftr"
        index_file_name = myPath + "preprocessed_files\\articles_index.pickle"
        results_file_name = myPath + "preprocessed_files\\rank.csv"

        with open(index_file_name, 'rb') as handle:
            index = pickle.load(handle)
        df = pd.read_feather(corpus_df_name, columns=None, use_threads=True)
        print("Data successfully uploaded")

        print("Building rank...")
        query = tokanize_text(query_sent, lower=True, remove_digits=True, is_lemmatized=True, remove_stop_words=True)
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
    tester.run_query(query_sent="What century did the Normans first gain their separate identity?")

if __name__ == "__main__":
    main()

