import os
from Query import Query
import nltk
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize as Tokenizer
import StringPreprocessingFunctions

class QueryLogger(object):
    def __init__(self):
        self.queries = {}
        self.list_of_all_queries = []

    def find_query(self, id):
        for x in self.queries: # Search queries for one that starts with our position
            if x.id == id:
                return x

    def add_query(self, query_obj): # Creates nested dictionary of a keyword + queries with that keyword
        for keyword in query_obj.keywords:

            if keyword not in self.queries: # Checks to see if query keyword exists and instantiates it if not
                self.queries[keyword] = {}
                self.queries[keyword][query_obj.query] = 1
            else:
                if query_obj.query not in self.queries[keyword]: #Checks if the query has already been added and instantiating it if not
                    self.queries[keyword][query_obj.query] = 1

                else: # If found, increment times queried
                    num_to_increment = self.queries[keyword][query_obj.query]
                    self.queries[keyword][query_obj.query] = num_to_increment + 1

    def load_and_write_aol_queries(self): # Loads AOL Queries from our program and loads all queries
        for file in os.listdir(f"AOL-Query-Logs\\"):
            read_file = open(f"AOL-Query-Logs\\{file}", 'r')
            print(f"-- Reading in {file} --")
            lines = read_file.readlines()
            for line in lines[1:len(lines)]: # Used to skip the header line

                line = line.replace("\n", "")
                info = line.split("\t") # Splits into info[0] = ID, info[1] = query, info[2] = datetime object
                self.list_of_all_queries.append(info[1])
                query_words = StringPreprocessingFunctions.preprocess_string(info[1])
                for query in query_words:
                    query = (Query(info[0], info[1], info[2]))
                    self.add_query(query)
        
        with open("query_log.json", "w") as outfile:
            json.dump(self.queries, outfile)

    def load_queries(self):
        with open('query_log.json') as json_file:
            self.queries = json.load(json_file)




     # def Save aol queries 
     # def load aol queries
     # return suggestions

#####################
# Used to run tests #
#####################

if __name__ == "__main__":

    # Start tests
    print("|>>> Printing Tests <<<|")
    print("------------------------")
    print("---- Instantiate QueryHolder")
    QueryLogger = QueryLogger()
    print("---- Add Query")
    QueryLogger.add_query(Query(1, "This is a query"))
    print("---- Load AOL Queries")
    QueryLogger.load_and_write_aol_queries()
    print("---- Print Queries")
    for query in QueryLogger.queries.keys:
        print(f"{query} | QueryHolder.queries[query]")
