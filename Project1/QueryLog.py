import datetime

class QueryObj(object):
    def __init__(self, id, query, timestamp = datetime.now()):
        self.id = id
        self.query = query
        self.timestamp = timestamp

    def print():
        print(f"Query ID: {self.id} | Timestamp: {self.timestamp")
        print(f"Query: {query}")




