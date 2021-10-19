import time
import pickle
import pandas as pd
import json
#from Project1.preprocessing import tokanize_text
#from Project1.GenerateSnippets_v2 import GenerateSnippets
#from Project1.QueryLogger import QueryLogger
from preprocessing import tokanize_text
from GenerateSnippets_v2 import GenerateSnippets
from QueryLogger import QueryLogger


class SearchEngineData():
    def __init__(self):
        self.query_logger = QueryLogger()


    def run_query(self, query_sent="No query submitted"):
        if query_sent == "No query submitted":
            return "No query submitted"
        print("Uploading data...")
        myp = ""

        # corpus_df_name = "preprocessed_files/wiki_df.ftr"         # TODO: https://drive.google.com/drive/folders/1rSeosE42x1R4yW014VgmFhJGA5Yj8qOr?usp=sharing
        # index_file_name = "preprocessed_files/wiki_index.pickle"  # TODO: Download file from google drive and move to the preprocessed_files folder in the Project1 directory
        # results_file_name = "preprocessed_files/rank.csv"

        ## Stephs filepaths
        #myp = "C:\\Users\\steph\\source\\repos\\info_retrieval\\Project1\\"
        #corpus_df_name = myp + "preprocessed_files\\wiki_df.ftr"
        #index_file_name = myp + "preprocessed_files\\wiki_index.pickle"
        #results_file_name = myp + "preprocessed_files\\rank.csv"

        corpus_df_name = "wiki_df.ftr"         # TODO: https://drive.google.com/drive/folders/1rSeosE42x1R4yW014VgmFhJGA5Yj8qOr?usp=sharing
        index_file_name = "wiki_index.pickle"  # TODO: Download file from google drive and move to the preprocessed_files folder in the Project1 directory
        results_file_name = "rank.csv"

        with open(index_file_name, 'rb') as handle:
            index = pickle.load(handle)
        df = pd.read_feather(corpus_df_name, columns=None, use_threads=True)
        print("Data successfully uploaded")

        print("Building rank...")
        query = tokanize_text(query_sent, lower=True, remove_digits=True, is_lemmatized=True, remove_stop_words=True)
        all_rank, rank = index.rank_docs(query, max_num=5)
        print(all_rank)
        print(rank)
        print("Rank is created")

        print("Generating snippets...")
        snippet_obj = GenerateSnippets(index.get_terms(), df)
        snippets_df = snippet_obj.getSnippets(query_sent, rank)
        #snippets_df.to_csv(results_file_name)

        snippets_json = json.dumps(snippets_df, indent = 4)

        print(snippets_json)

        return snippets_json


def main():
    tester = SearchEngineData()
    tester.run_query(query_sent="Boise in Idaho")

if __name__ == "__main__":
    main()

