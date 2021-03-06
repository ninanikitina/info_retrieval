from nltk.tokenize import RegexpTokenizer
import pandas as pd
import csv
import time
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus.reader.wordnet import VERB, NOUN, ADJ, ADV
from nltk.tag import StanfordPOSTagger
from nltk.corpus import stopwords
from nltk import word_tokenize
from tqdm import tqdm
from functools import lru_cache
from collections import defaultdict
import matplotlib.pyplot as plt
import pickle
from Project1.IndexTerm import IndexTerm
import json
from random import randrange

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
tokenizer = RegexpTokenizer(r'\w+')
ll = lru_cache(maxsize=50000)(lemmatizer.lemmatize) # this will help to speed up lemmatizing process


def clean_df(tokenized_df, dictionary, min_frequency_term, tokenized_clean_df_name):
    """
    Remove from the the index all words that appears less then 5 times and
    stop words ['also', 'first', 'one', 'new', 'two']
        Parameters
        ----------
        tokenized_df : pandas DataFrame
            Tokenized DataFrame to clean
        dictionary : dict
            Term frequency in tokenized_df
        min_frequency_term: int
            The minimum frequency of term to keep it in index
    """
    content = tokenized_df["all_content"].to_list()
    id = tokenized_df["id"].to_list()
    clean_df = pd.DataFrame(columns=["all_content", "id"])
    term_to_remove = set(k for k, v in dictionary.items() if v < min_frequency_term)
    most_frequent_words = {'also', 'first', 'one', 'new', 'two'}
    term_to_remove.update(most_frequent_words)
    for i, terms in enumerate(tqdm(content)):
        updated_tokens = [word for word in terms if word not in term_to_remove]
        clean_df = clean_df.append({'all_content': updated_tokens, 'id': id[i]}, ignore_index=True)
    clean_df.to_feather(tokenized_clean_df_name)
    return clean_df

def chose_questions(filename, df_name):
    df = pd.DataFrame(columns=["question", "answer"])
    with open(filename) as f:
        data = json.load(f)
    for i, article in enumerate(data.get("data")):
        if i == 439:
            break
        is_impossible = True
        paragraph_num = len(article.get('paragraphs'))
        if paragraph_num == 0:
            continue
        while (is_impossible):
            chosen_paragraph = randrange(paragraph_num)
            question_num = len(article.get('paragraphs')[chosen_paragraph].get('qas'))
            if question_num == 0:
                continue
            chosen_question = randrange(question_num)
            is_impossible = article.get('paragraphs')[chosen_paragraph].get('qas')[chosen_question].get('is_impossible')
        question = article.get('paragraphs')[chosen_paragraph].get('qas')[chosen_question].get('question')
        answer = article.get('paragraphs')[chosen_paragraph].get('qas')[chosen_question].get('answers')[0].get('text')
        df = df.append({'question': question, 'answer': answer}, ignore_index=True)
        print(i)
    df.to_feather(df_name)
    return df



def read_corpus(filename_1, filename_2, df_file_name):
    """
    Read the corpus data and convert unprocessed text to data frame.
    In the original paper it says 536 articles however data set has only 477
        Parameters
        ----------
        filenames : str
            The name of json files
        df_file_name : str
            The name of data frame file to save to
    """
    df = pd.DataFrame(columns=["content", "title", "id"])
    with open(filename_1) as f:
        data = json.load(f)
    df = update_content_df(df, data)

    with open(filename_2) as f:
        data = json.load(f)
    df = update_content_df(df, data)
    df.to_feather(df_file_name)
    return df

def get_rank_frequency(dictionary, constant):
    """
    Build Zipf's law
        Parameters
        ----------
        dictionary : dictionary
            {term: frequency}
        constant : int
            Zipf's law constant
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
    plt.scatter(x = df_rank['rank'], y = df_rank['frequency'], marker='o', label = "Word frequency in Stanford article set")
    zipf_law = constant / df_rank['rank']
    plt.plot(df_rank['rank'], zipf_law, label = "Zipf law", color = 'red')
    plt.gca().set_yscale('log')
    # plt.gca().set_xscale('log')
    plt.legend()
    plt.title("Zipf law Stanford's articles collection")
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


def tokanize_text(text, lower, remove_digits, is_lemmatized, remove_stop_words):
    # tokanize content and title of each document in the corpus, remove punctuation,
    # lower case all words if lower is True or keep original case otherwise
    tokens = get_tokens(text, remove_digits, remove_stop_words)

    if lower:
        tokens = [word.lower() for word in tokens]

    # Lemmatized words
    if is_lemmatized:
        # tokens = ll(tokens)
        # temp = []
        # for word in tokens:
        #     temp.extend(ll(word))
        tokens = [ll(word) for word in tokens]

    return tokens

def get_tokens(query, remove_digits, remove_stop_words):
    tokens = []
    if isinstance(query, str):
        tokens.extend(tokenizer.tokenize(query))

    # remove stop words
    if remove_stop_words:
        tokens = [word for word in tokens if word not in stop_words]

    # Remove digits
    if remove_digits:
        tokens = [word for word in tokens if not word.isdigit()]
    return tokens

def tokenize_query(query, lower, remove_digits, is_lemmatized, remove_stop_words):

    """ Preprocesses query (question) and returns dictionary   
    : lower case all words if lower is True or keep original case
    : calls queryTag() to get dictionary of tokens and whether or not they are a noun 
    : compares tokens from preprocessed query (question) and compares it to keys in dictionary
    : returns a dictionary of tokens from preprocessed query as key and True or False as value --> {Noun: True}
    """
    tokens = get_tokens(query, remove_digits, remove_stop_words)
    dict = queryTag(tokens)

    # Lemmatized words
    if is_lemmatized:
        newDict = {}
        for key, value in dict.items():
            newDict[ll(key)] = value

    return newDict


def queryTag(query):
    """ Takes query and determines if each token is a noun or not   
    : lower case all words
    : uses nltk pos_tag to determine if word is a noun or not
    : creates dictionary using words from query (question) as keys and value of True or False 
    : returns the dictionary --> {Noun: True}
    """

    # query1 = word_tokenize(query)
    queryList = nltk.pos_tag(query)
    dict = {}

    for w in queryList:
        proc_word = w[0].lower()
        #proc_word = ll(proc_word)
        dict.setdefault(proc_word, False)
        if w[1].startswith("NNP"):
            dict[proc_word] = True

    return dict


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
    content = df["content"].to_list()
    title = df["title"].to_list()
    id = df["id"].to_list()

    for i, text in enumerate(tqdm(content)):
        tokens = []
        # tokanize content and title of each document in the corpus, remove punctuation,
        # lower case all words if lower is True or keep original case otherwise
        if isinstance(text, str):
            tokens.extend(tokenizer.tokenize(text.lower() if lower else text))
        if isinstance(title[i], str):
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

def update_content_df(df, data):
        for i, article in enumerate(data.get("data")):
            content = ""
            for paragraph in article.get('paragraphs'):
                content = content + paragraph.get("context") + "\n"
            title = article.get("title")
            df = df.append({'content': content, 'title': title, 'id': i}, ignore_index=True)
        return df

def main():
    read_new_corpus = True
    tokenize_new_corpus = True
    create_new_dictionary = True
    clean_tokenized_corpus = True
    #myPath = "C:\\Users\\steph\\source\\repos\\info_retrieval\\Project2\\"
    find_test_questions = True
    myPath = ""
    MIN_WORD_FREQ = 3

    # Wiki collection for main project
    json_file_name_1 = myPath + "preprocessed_files\\dev-v2.0.json" 
    json_file_name_2 = myPath + "preprocessed_files\\train-v2.0.json"
    corpus_df_name = myPath + "preprocessed_files\\article_df.ftr"

    dictionary_file_name = myPath +  "preprocessed_files\\articles_dictionary.pickle"
    tokenized_df_name = myPath + "preprocessed_files\\articles_tokenized_df.ftr"
    tokenized_clean_df_name = myPath + "preprocessed_files\\articles_tokenized_clean_df.ftr"
    index_file_name = myPath + "preprocessed_files\\articles_index.pickle"
    questions_df_name = myPath + "preprocessed_files\\questions_df.ftr"


    if read_new_corpus:
        df = read_corpus(filename_1=json_file_name_1, filename_2=json_file_name_2, df_file_name=corpus_df_name)
        print(df)
    if find_test_questions:
        df_questions_answers = chose_questions(filename=json_file_name_2, df_name=questions_df_name)


    if tokenize_new_corpus:
        # Tokenize corpus or download previously tokenized
        if df is None:
            df = pd.read_feather(corpus_df_name, columns=None, use_threads=True)

        tokenized_df = tokanize_corpus(df=df, lower=True, remove_digits=True,
                                   is_lemmatized=True, remove_stop_words=True,
                                    df_file_name=tokenized_df_name)
    else:
        tokenized_df = pd.read_feather(tokenized_df_name, columns=None, use_threads=True)

    if create_new_dictionary:
        dictionary = create_term_frequency_index(tokenized_df)
        with open(dictionary_file_name, 'wb') as handle:
            pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(dictionary_file_name, 'rb') as handle:
            dictionary = pickle.load(handle)

    # check the number of words in the index
    print("The size of index before cleaning is :" + str(len(dictionary)))
    d_five_and_more = dict((k, v) for k, v in dictionary.items() if int(v) >= MIN_WORD_FREQ)
    print("The size of index where all terms appears more than "+ str(MIN_WORD_FREQ)+" times :" + str(len(d_five_and_more)))
    get_rank_frequency(dictionary, 100000)   # print zipf's low

    if clean_tokenized_corpus:
        tokenized_clean_df = clean_df(tokenized_df, dictionary, MIN_WORD_FREQ, tokenized_clean_df_name)
    else:
        tokenized_clean_df = pd.read_feather(tokenized_clean_df_name, columns=None, use_threads=True)
    clean_dictionary = create_term_frequency_index(tokenized_clean_df)
    print("The size of index after cleaning is :" + str(len(clean_dictionary)))

    index = IndexTerm(tokenized_clean_df)
    with open(index_file_name, 'wb') as handle:
        pickle.dump(index, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()