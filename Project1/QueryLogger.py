import os
from Query import Query
import json
import StringPreprocessingFunctions
import time

class QueryLogger(object):
    def __init__(self):
        self.queries = self.load_queries()
        self.list_of_all_queries = []

    def add_query(self, query, id=-124, datetime=time.time()): # Creates nested dictionary of a keyword + queries with that keyword
        if id == -124:
            id = len(self.list_of_all_queries)
        query_obj = Query(id, query, datetime)
        self.list_of_all_queries.append(query_obj)
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
                self.add_query(info[1], id = info[0], datetime=info[2]) # Add query, ID and Datetime and let add_query split/add the query
        
                with open("query_log.json", "w") as outfile:
                    json.dump(self.queries, outfile)

    def load_queries(self):
        with open('query_log.json') as json_file:
             self.queries = json.load(json_file)



    def get_suggestions(self, query):
        # Split query into individual words and perform string preprocessing
        words = query.split()
        words = StringPreprocessingFunctions.preprocess_string(words)
        all_queries = []
        best_queries = []
        good_queries = []

        # Go through each word in a query
        for word in words:
            # Check the queries searched based on the weird
            if word in self.queries:
                related_queries = self.queries[word].keys()
                # Find all queries associated with that word
                for query in related_queries:
                    all_queries.append((query, self.queries[word][query]))

        # For each query, check if all query words are found in that query
        for query in all_queries:
            if all(word in query[0] for word in words):
                best_queries.append(query)
            else:
                if any(word in query[0] for word in words):
                    good_queries.append(query)

                # If we found high quality queries, return higher quality queries, else return the highest frequency searches based on the original query
        if len(best_queries) > 0:
            best_queries.sort(reverse=True, key=tuple_sort)
            best_queries = best_queries[:5]
            retVal = {}
            retVal["Suggestions"] = remove_tuples_and_jsonify(best_queries)
            retVal["ResultLength"] = len(best_queries)
            retVal = json.dumps(retVal)
            return retVal
        else:
            good_queries.sort(reverse=True, key=tuple_sort)
            good_queries = good_queries[:5]
            retVal = {}
            retVal["Suggestions"] = remove_tuples_and_jsonify(good_queries)
            retVal["ResultLength"] = len(good_queries)
            retVal = json.dumps(retVal)
            return retVal
        

def tuple_sort(tuple):
    return tuple[1]

def remove_tuples_and_jsonify(tuple_list):
    return_val = {}
    counter = 0
    for tuple in tuple_list:
        return_val[counter] = tuple[0]
        counter += 1
    return return_val
        

#####################
# Used to run tests #
#####################

if __name__ == "__main__":

     # Start tests
    print("|>>> Printing Tests <<<|")
    print("------------------------")
    print("---- Instantiate QueryHolder")
    QueryLogger = QueryLogger()
    print("---- Load AOL Queries")
    QueryLogger.load_queries()
    QueryLogger.load_and_write_aol_queries()
    print("---- Print Queries")
    queries = QueryLogger.queries.keys()
    for query in queries:
         values = list(QueryLogger.queries[query].values())
         keys = list(QueryLogger.queries[query].keys())
         for x in range(0, len(keys)):
            print(f"Query Word: {query} | Associated Query: {keys[x]} | Frequency: {values[x]}")