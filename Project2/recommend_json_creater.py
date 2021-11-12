import json
import os
import StringPreprocessingFunctions as preprocessor

class QueryCreator():

    def __init__(self):
        self.recommendations = {
            "topics": {}, 
            "trigrams": {}, 
            "bigrams":{}, 
            "unigrams":{}
            }

    def create_topics(self): # Loads AOL Queries from our program and loads all queries

        for file in os.listdir(f"Project2\\preprocessing_files"):
            read_file = open(f"Project2\\preprocessing_files\\{file}", 'r')
            print(f"-- Reading in {file} --")
            json_data = json.load(read_file)
            paragraphs = json_data["data"] # Pull data from the json
            for questions in paragraphs: # Grab each query in Data list
                for qas in questions["paragraphs"]: # Look at the paragraphs
                    for question in qas['qas']: # Go through each 'qas'
                        query_processed = preprocessor.preprocess_string(question['question'])
                        for noun in query_processed['nouns']:
                            self.process_query(noun, query_processed['original_query'])
                        

        with open(f"Project2\\recommendations.json", "w") as outfile:
            json.dump(self.recommendations, outfile, sort_keys=True, indent=4)        

    def process_query(self, noun, query, query_type='topics'):
        if noun not in self.recommendations[query_type]: # Checks to see if query keyword exists and instantiates it if not
                self.recommendations[query_type][noun]= {}
                self.recommendations[query_type][noun][query] = 1
        else:
            if query not in self.recommendations[query_type][noun]: #Checks if the query has already been added and instantiating it if not
                self.recommendations[query_type][noun][query] = 1
            else:
                self.recommendations[query_type][noun][query] += 1

    def create_ngrams(self):
        for file in os.listdir(f"Project2\\preprocessing_files"):
            read_file = open(f"Project2\\preprocessing_files\\{file}", 'r')
            print(f"-- Reading in {file} --")
            json_data = json.load(read_file)
            paragraphs = json_data["data"] # Pull data from the json
            for questions in paragraphs: # Grab each query in Data list
                for qas in questions["paragraphs"]: # Look at the paragraphs
                    for question in qas['qas']: # Go through each 'qas'
                        words = question['question'].split()
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
            with open(f"Project2\\recommendations.json", "w") as outfile:
                    json.dump(self.recommendations, outfile, sort_keys=True, indent=4)   

        with open(f"Project2\\recommendations.json", "w") as outfile:
                    json.dump(self.recommendations, outfile, sort_keys=True, indent=4)


    def load_recommendations(self):
        f = open(f"Project2\\recommendations-topics.json")
        self.recommendations = json.load(f)

if __name__ == "__main__":
    x = QueryCreator()
    x.load_recommendations()
    x.create_ngrams()
