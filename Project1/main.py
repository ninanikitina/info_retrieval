import time
import pickle
import pandas as pd
from Project1.preprocessing import tokanize_text
from Project1.GenerateSnippets_v2 import GenerateSnippets




def main():
    #  Wiki collection for project
    corpus_df_name = "preprocessed_files/wiki_df.ftr"         # TODO: https://drive.google.com/drive/folders/1rSeosE42x1R4yW014VgmFhJGA5Yj8qOr?usp=sharing
    index_file_name = "preprocessed_files/wiki_index.pickle"  # TODO: Download file from google drive and move to the preprocessed_files folder in the Project1 directory


    #  Disney collection for test
    # index_file_name = "preprocessed_files/disney_index.pickle"  # TODO: to create disney_index.pickle file run preprocessing script
    # corpus_df_name = "preprocessed_files/disney_dataframe.ftr"

    results_file_name = "preprocessed_files/rank.csv"
    print("Loading index...")
    start = time.time()
    with open(index_file_name, 'rb') as handle:
        index = pickle.load(handle)
    end = time.time()
    print(f"Loading time is ", (end - start))

    df = pd.read_feather(corpus_df_name, columns=None, use_threads=True)

    query_sent = "the dog jumped high"  # Test query parsing
    query = tokanize_text(query_sent, lower=True, remove_digits=True, is_lemmatized=True, remove_stop_words=True)
    rank = index.rank_docs(query, max_num=5)
    print(f"Preprocessed query ", query)
    print(f"The highest rated documents are ", rank)

    a = index.get_terms()
    snippet_obj = GenerateSnippets(index.get_terms(), df)

    snippets_df = snippet_obj.getSnippets(query_sent, rank)

    snippets_df.to_csv(results_file_name)

    print(snippets_df)


if __name__ == "__main__":
    main()
