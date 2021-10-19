import datetime
import StringPreprocessingFunctions

class Query(object):
    def __init__(self, id, query, timestamp = datetime.time()):
        self.id = id
        self.query = query
        self.timestamp = timestamp
        self.keywords = StringPreprocessingFunctions.preprocess_string(query) # Could use updating

    def print(self):
        print(f"Query ID: {self.id} | Timestamp: {self.timestamp}")
        print(f"Query: {self.query}")
        print("--------------------------------------------------")




