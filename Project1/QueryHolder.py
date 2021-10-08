import os
from QueryLog import Query
import nltk
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize as Tokenizer

class QueryHolder(object):
    def __init__(self):
        self.queries = {}

    def find_query(self, id):
        for x in self.queries: # Search queries for one that starts with our position
            if x.id == id:
                return x

    def add_query(self, query_obj): # Creates nested dictionary of a keyword + queries with that keyword
        if query_obj.keyword not in self.queries: # Checks to see if query keyword exists and instantiates it if not
            self.queries[query_obj.keyword] = {}
            self.queries[query_obj.keyword][query_obj.query] = 1
        else:
            if query_obj.query not in self.queries[query_obj.keyword]: #Checks if the query has already been added and instantiating it if not
                self.queries[query_obj.keyword][query_obj.query] = 1

            else: # If found, increment times queried
                num_to_increment = self.queries[query_obj.keyword][query_obj.query]
                self.queries[query_obj.keyword][query_obj.query] = num_to_increment + 1

    def load_aol_queries(self): # Loads AOL Queries from our program and loads all queries
        for file in os.listdir(f"AOL-Query-Logs\\"):
            read_file = open(f"AOL-Query-Logs\\{file}", 'r')
            print(f"-- Reading in {file} --")
            lines = read_file.readlines()
            for line in lines[1:len(lines)]: # Used to skip the header line
                text_tokens = Tokenizer(line)
                tokens_without_stopwords = [word for word in text_tokens if not word in stopwords.words()]
                line = line.replace("\n", "")
                info = line.split("\t")
                query_words = info[1].split()
                for query in query_words:
                    query = (Query(info[0], info[1], info[2]))
                    self.add_query(query)
                

#####################
# Used to run tests #
#####################

if __name__ == "__main__":

    # Start tests
    print("|>>> Printing Tests <<<|")
    print("------------------------")
    print("---- Instantiate QueryHolder")
    QueryHolder = QueryHolder()
    print("---- Add Query")
    QueryHolder.add_query(Query(1, "This is a query"))
    print("---- Load AOL Queries")
    QueryHolder.load_aol_queries()
    print("---- Print Queries")
    for query in QueryHolder.queries.keys:
        print(f"{query} | QueryHolder.queries[query]")

