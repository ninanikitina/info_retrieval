import time
import pickle
from Project1.preprocessing import tokanize_text


def main():
    #  main file of the project
    # index_file_name = "preprocessed_files/wiki_index.pickle"  # TODO: Download file from google drive and move to the preprocessed_files folder in the Project1 directory
                                                                # TODO: https://drive.google.com/drive/folders/1rSeosE42x1R4yW014VgmFhJGA5Yj8qOr?usp=sharing
    #  Disney collection for test
    index_file_name = "preprocessed_files/disney_index.pickle"  # TODO: to create disney_index.pickle file run preprocessing script

    print("Loading index...")
    start = time.time()
    with open(index_file_name, 'rb') as handle:
        index = pickle.load(handle)
    end = time.time()
    print(f"Loading time is ", (end - start))

    query_sent = "bear and raccoon are not friends,.. and hfisdniu"  # Test query parsing
    query = tokanize_text(query_sent, lower=True, remove_digits=True, is_lemmatized=True, remove_stop_words=True)
    rank = index.rank_docs(query, max_num=5)
    print(f"Preprocessed query ", query)
    print(f"The highest rated documents are ", rank)


if __name__ == "__main__":
    main()
