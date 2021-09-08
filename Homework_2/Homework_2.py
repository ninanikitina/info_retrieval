from nltk import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
from nltk.util import ngrams
import matplotlib.pyplot as plt


def add_tokens(tokens, texts):
    for text in texts:
        tokens.extend(word_tokenize(text.lower()))
    return tokens


def clean_tokens(tokens, include_stop_words):
    stop_words = set(stopwords.words('english'))

    # remove stop words
    if not include_stop_words:
        tokens = [word for word in tokens if word not in stop_words]

    # remove punctuation
    tokens = [word for word in tokens if word.isalnum()]
    return tokens


def get_rank_frequency(tokens, n, constant):
    n_grams = ngrams(tokens, n)
    df_rank = pd.Series(n_grams).value_counts().rename_axis(str(n) + "-grams").to_frame('frequency')

    # print the top 5 n-grams
    print(df_rank[:5])

    # plot rank frequency curve
    df_rank['rank'] = list(range(1, len(df_rank.index) + 1, 1))
    plt.scatter(x = df_rank['rank'], y = df_rank['frequency'], marker='o', label = "Word frequency in Disney")
    zipf_law = constant / df_rank['rank']
    plt.plot(df_rank['rank'], zipf_law, label = "Zipf low", color = 'red')
    plt.gca().set_yscale('log')
    plt.gca().set_xscale('log')
    plt.legend()
    plt.title(str(n) + "-grams")
    plt.show()


def main():
    include_stop_words = False
    df = pd.read_csv("disney_plus_shows.csv", usecols=["title", "plot"])
    titles = df["title"].dropna().to_list()
    plots = df["plot"].dropna().to_list()

    # # Zipf’s Law for text in titel and plot columns
    tokens = []
    tokens = add_tokens(tokens, titles)
    tokens = add_tokens(tokens, plots)
    tokens = clean_tokens(tokens, include_stop_words)

    get_rank_frequency(tokens, 1, constant = 800)
    get_rank_frequency(tokens, 2, constant = 100)

    # Heaps’ Law for text in titel and plot columns
    dictionary = set()
    vocabulary_size = []
    words_number = []
    all_tokens = []
    for idx, plot in enumerate(plots):
        tokens = word_tokenize(plot.lower())
        tokens.extend(word_tokenize(titles[idx].lower()))
        # Count all words
        tokens = clean_tokens(tokens, include_stop_words)
        all_tokens.extend(tokens)
        for token in tokens:
            dictionary.add(token)
        vocabulary_size.append(len(dictionary))
        words_number.append(len(all_tokens))

    heaps_law = []
    for idx, plot in enumerate(vocabulary_size):
        heaps_law.append(3 * (words_number[idx])**0.78)

    plt.plot(words_number, vocabulary_size, color = "black", label = "Disney collection")
    plt.plot(words_number, heaps_law, color = "red", label = "Heap law")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
