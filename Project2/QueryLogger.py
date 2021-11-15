import os
import json
import StringPreprocessingFunctions
import time
import asyncio
import StringPreprocessingFunctions as preprocessor

class QueryLogger(object):
    def __init__(self):
        self.recommendations = self.load_queries()
        self.list_of_all_queries = []

    def load_queries(self):
        with open(f'Project2\\recommendations.json') as json_file:
             self.recommendations = json.load(json_file)
    
    async def save_queries(self):
        with open(f"Project2\\recommendations.json", "w") as outfile:
            asyncio.create_task(json.dump(self.recommendations, outfile, sort_keys=True, indent=4))

    async def update_recommendations(self, query):
        topics = asyncio.create_task(self.add_topics(query)) 
        ngrams = asyncio.create_task(self.add_ngrams(query)) 
        await topics
        await ngrams
    
    async def add_topics(self, query): # Creates nested dictionary of a keyword + queries with that keyword
        query_processed = preprocessor.preprocess_string(query)
        for noun in query_processed['nouns']:
            self.process_query(noun, query_processed['original_query'], query_type="topics")

    async def add_ngrams(self, query):
            words = query.split()
            for x in range(0, len(words)):
                if (x+3) < len(words):
                    trigram = words[x] + " " + words[x+1] + " " + words[x+2]
                    self.process_query(trigram.lower(), words[x+3], "trigrams")
                    if (x+4) < len(words):
                        self.process_query(trigram.lower(), words[x+3] + " " + words[x+4], "trigrams")
                if (x+2) < len(words):
                    bigram = words[x] + " " + words[x+1]
                    self.process_query(bigram.lower(), words[x+2], "bigrams")
                    if (x+3) < len(words):
                        self.process_query(bigram.lower(), words[x+2] + " " + words[x+3], "bigrams")
                if (x+1) < len(words):
                    unigram = words[x]
                    self.process_query(unigram.lower(), words[x+1], "unigrams")
                    if (x+2) < len(words):
                        self.process_query(unigram.lower(), words[x+1] + " " + words[x+2], "unigrams")
                    if (x+3) < len(words):
                        self.process_query(unigram.lower(), words[x+1] + " " + words[x+2] + " " + words[x+3], "unigrams")
            
        
    def process_query(self, keyword, query, query_type='topics'):
        if keyword not in self.recommendations[query_type]: # Checks to see if query keyword exists and instantiates it if not
                self.recommendations[query_type][keyword]= {}
                self.recommendations[query_type][keyword][query] = 1
        else:
            if query not in self.recommendations[query_type][keyword]: #Checks if the query has already been added and instantiating it if not
                self.recommendations[query_type][keyword][query] = 1
            else:
                self.recommendations[query_type][keyword][query] += 1

    def get_suggestions(self, query):
        # Split query into individual words and perform string preprocessing
        words = query.lower().split()
        
        # Create lists to use for future recommendation filtering
        unigram_list = []
        bitrigram_list = []
        topic_list = []

        # If query is long enough, create unigrams/bigrams/trigrams and add known searches to the lists
        if len(words) >= 3:
            for x in range (0, len(words) -3):
                trigram = words[x] + " " + words[x+1] + " " +  words[x+2]
                if trigram in self.recommendations['trigrams']:
                    for key in self.recommendations['trigrams'][trigram].keys():
                        bitrigram_list.append((key, self.recommendations['trigrams'][trigram][key]))
                
                bigram = words[x] + " " +  words[x+1]
                if bigram in self.recommendations['bigrams']:
                    for key in self.recommendations['bigrams'][bigram].keys():
                        bitrigram_list.append((key, self.recommendations['bigrams'][bigram][key]))
                
                unigram = words[x]
                if unigram in self.recommendations['unigrams']:
                    for key in self.recommendations['unigrams'][unigram].keys():
                        unigram_list.append((key, self.recommendations['unigrams'][unigram][key]))
                        
                        
        # If query is less than 3, check for bigrams and unigrams only
        elif len(words) < 3:
            for x in range (0, len(words)):
                unigram = words[x]
                if x+1 == len(words):
                    bigram = ""
                else:
                    bigram = words[x] + " " +  words[x+1]

                if bigram in self.recommendations['bigrams']:
                    for key in self.recommendations['bigrams'][bigram].keys():
                        bitrigram_list.append((key, self.recommendations['bigrams'][bigram][key]))
                            
                if unigram in self.recommendations['unigrams']:
                    for key in self.recommendations['unigrams'][unigram].keys():
                        unigram_list.append((key, self.recommendations['unigrams'][unigram][key]))

        # Go through each word in the query to search for topics
        for word in words:
            # Check if there is a topic for each word
            if word in self.recommendations['topics']:
                for key in self.recommendations['topics'][word].keys():
                    topic_list.append((word, key, self.recommendations['topics'][word][key]))

        # For each list, sort so the most frequently are at the top
        topic_list.sort(reverse=True, key=tuple_sort)
        bitrigram_list.sort(reverse=True, key=tuple_sort)
        unigram_list.sort(reverse=True, key=tuple_sort)

        retVal = set() # Ensure no duplicates by making this a set
        while ((len(topic_list) != 0 or len(topic_list) != 0 or len(unigram_list) != 0) and (len(retVal) <= 5)): # Loop through until RetVal has 5 values or there are no more values left to loop through. Ratio is for 1 unigram/2 bigrams or trigrams/ 3 topics ideally
            if len(unigram_list) > 0:
                retVal.add(unigram_list.pop(0)[0])
            if len(bitrigram_list) > 0:
                retVal.add(bitrigram_list.pop(0)[0])
            if len(bitrigram_list) > 0:
                retVal.add(bitrigram_list.pop(0)[0])
            if len(topic_list) > 0:
                retVal.add(topic_list.pop(0)[1])
            if len(topic_list) > 0:
                retVal.add(topic_list.pop(0)[1])
            if len(topic_list) > 0:
                retVal.add(topic_list.pop(0)[1])
        
        # If no thing found, return that no results were found
        if len(retVal) == 0:
            retVal = ['','','','','']  
        # Else, set retVal to a list so it can be sorted by length and returned shortest recommendation to longest recommendation
        else:
           retVal =  list(retVal)
           retVal.sort(key=len)
        
        # Create a JSON string for pushing to the website
        retVal = json.dumps({"recommendations": retVal})
        return retVal

def tuple_sort(tuple):
    if len(tuple) == 3:
        return tuple[2]
    else:
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
    x=QueryLogger()
    x.load_queries()
    x.get_suggestions("When did the")

  