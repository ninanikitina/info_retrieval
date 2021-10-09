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


lemmatizer = WordNetLemmatizer()
ll = lru_cache(maxsize=50000)(lemmatizer.lemmatize) # this will help to speed up lemmatizing process


def read_corpus(filename, df_file_name):
    """
    Read the corpus data and convert unprocessed text to data frame
        Parameters
        ----------
        filename : str
            The name of csv file
        df_file_name : str
            The name of data frame file to save to
    """
    df = pd.read_csv(filename, usecols=["content", "title", "id"])
    df.to_feather(df_file_name)
    return df

def get_rank_frequency(dictionary, constant):
    """
    Build Zipf's low
        Parameters
        ----------
        dictionary : dictionary
            {term: frequency}
        constant : int
            Zipf's low constant
    """
    frequencies = list(dictionary.values())
    sorted_frequencies = sorted(frequencies, reverse=True)
    df_rank = pd.Series(sorted_frequencies).to_frame('frequency')

    # print the top 5 words and their frequency
    print("***********************************")
    print(sorted(dictionary, key=dictionary.get, reverse=True)[:5])
    print(df_rank[:5])

    # plot rank frequency curve
    df_rank['rank'] = list(range(1, len(df_rank.index) + 1, 1))
    plt.scatter(x = df_rank['rank'], y = df_rank['frequency'], marker='o', label = "Word frequency in Wiki")
    zipf_law = constant / df_rank['rank']
    plt.plot(df_rank['rank'], zipf_law, label = "Zipf low", color = 'red')
    plt.gca().set_yscale('log')
    # plt.gca().set_xscale('log')
    plt.legend()
    plt.title("fr")
    plt.show()


def create_term_frequency_index(df):
    """
    Creates dictionary {term: frequency} based on tokens
        Parameter
        ----------
        df : DataFrame
    """
    dictionary = defaultdict(int)
    for document_words in tqdm(df['all_content']):
        for word in document_words:
            dictionary[word] += 1
    return dictionary


def lemmatize (tokens):
    """
    TODO: Not use in this Project_1
    To make Lemmatization process more efficient this example were used:
    https://towardsdatascience.com/building-a-text-normalizer-using-nltk-ft-pos-tagger-e713e611db8
    Though this way of lemmatizing is more thoughtful, it is more time consuming, ~20 times more
    """
    dict_pos_map = {
        # Look for NN in the POS tag because all nouns begin with NN
        'NN': NOUN,
        # Look for VB in the POS tag because all nouns begin with VB
        'VB': VERB,
        # Look for JJ in the POS tag because all nouns begin with JJ
        'JJ': ADJ,
        # Look for RB in the POS tag because all nouns begin with RB
        'RB': ADV
    }
    st = StanfordPOSTagger(
        model_filename="stanford/stanford-postagger-full-2020-11-17/models/english-bidirectional-distsim.tagger",
        path_to_jar="stanford/stanford-postagger-full-2020-11-17/stanford-postagger.jar")
    lemmatized_tokens = []
    for tagged_word in st.tag(tokens):
        temp = tagged_word[0]
        if tagged_word[1][:2] in dict_pos_map.keys():
            temp = ll(tagged_word[0],pos=dict_pos_map[tagged_word[1][:2]])
        lemmatized_tokens.append(temp)
    return lemmatized_tokens


def tokanize_corpus(df, lower, remove_digits, is_lemmatized, remove_stop_words, df_file_name):
    """
    Tokenize "content" and "title" columns of corpus, based on provided flags
    and saves results as a DataFrame with two colmns:
        "all_content" list of all tokens for particular document
        "id" id of the particular document
    Parameters
        ----------
        filename : df
            The table (dataframe)
        remove_digits: boolean
            Flag - True if digits should be removed
        is_lemmatized: boolean
            Flag - True if terms should be lemmatized
        remove_stop_words: boolean
            Flag - True if standart stop words should be removed
        df_file_name: str
            File path to save document
    """
    clean_df = pd.DataFrame(columns=["all_content", "id"])
    tokenizer = RegexpTokenizer(r'\w+')
    content = df["content"].dropna().to_list()
    title = df["title"].dropna().to_list()
    id = df["id"].dropna().to_list()

    for i, text in enumerate(tqdm(content)):
        tokens = []
        # tokanize content and title of each document in the corpus, remove punctuation,
        # lower case all words if lower is True or keep original case otherwise
        tokens.extend(tokenizer.tokenize(text.lower() if lower else text))
        tokens.extend(tokenizer.tokenize(title[i].lower() if lower else title[i]))

        stop_words = set(stopwords.words('english'))

        # remove stop words
        if remove_stop_words:
            tokens = [word for word in tokens if word not in stop_words]

        # Remove digits
        if remove_digits:
            tokens = [word for word in tokens if not word.isdigit()]

        # Lemmatized words
        if is_lemmatized:
            # tokens = lemmatize(tokens)
            temp = []
            for word in tokens:
                temp.extend(ll(word))
        clean_df = clean_df.append({'all_content': tokens, 'id': id[i]}, ignore_index=True)
    clean_df.to_feather(df_file_name)
    return clean_df


def main():
    corp_file_nme = "../project_1_Wiki_sample.csv"
    read_new_corpus = False  # False saves 20 sec. True if csv file should be read to create a data base, False if previously created data frame should be download
    tokenize_new_corpus = False  # False saves 10 hours. True if need to preprocess new text corpus. Preprocessing takes from 10 to 12 hours
    read_new_dictionary = False  # False saves 10 seconds.
    dictionary_file_name = "preprocessed_files/dictionary.pickle"
    corpus_df_name = "preprocessed_files/dataframe.ftr"
    tokenized_df_name = "preprocessed_files/tokenized_dataframe.ftr"

    df = None
    if read_new_corpus:
        # Read csv file and save it to data frame format
        df = read_corpus(filename=corp_file_nme, df_file_name=corpus_df_name)

    if tokenize_new_corpus:
        # Tokenize corpus or download previously tokenized
        if df is None:
            df = pd.read_feather(corpus_df_name, columns=None, use_threads=True)

        tokenized_df = tokanize_corpus(df=df, lower=True, remove_digits=True,
                                   is_lemmatized=True, remove_stop_words=True,
                                    df_file_name=tokenized_df_name)
    else:
        tokenized_df = pd.read_feather(tokenized_df_name, columns=None, use_threads=True)

    if read_new_dictionary:
        dictionary = create_term_frequency_index(tokenized_df)
        with open(dictionary_file_name, 'wb') as handle:
            pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(dictionary_file_name, 'rb') as handle:
            dictionary = pickle.load(handle)

    # check the number of words in the index
    print("The size of index is :" + str(len(dictionary)))
    d_five_and_more = dict((k, v) for k, v in dictionary.items() if int(v) >= 5)
    print("The size of index where all terms appears more than five times :" + str(len(d_five_and_more)))
    get_rank_frequency(dictionary, 1000000)   # print zipf's low


if __name__ == "__main__":
    main()
