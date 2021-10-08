import os
from QueryLog import Query
import pandas as pd

class QueryHolder(object):
    def __init__(self):
        self.queries = []

    def find_query(self, id):
        for x in self.queries: # Search queries for one that starts with our position
            if x.id == id:
                return x

    def add_query(self, QueryObj): # Adds QueryObj
        self.queries.add(QueryObj)

    def load_aol_queries(self):
        for file in os.listdir(f"AOL-Query-Logs\\"):
            read_file = open(f"AOL-Query-Logs\\{file}", 'r')
            print(f"-- Reading in {file} --")
            lines = read_file.readlines()
            for line in lines[1:len(lines)]:
                info = line.split("\t")
                self.queries.append(Query(info[0], info[1], info[2]))


if __name__ == "__main__":
    QueryHolder = QueryHolder()
    QueryHolder.load_aol_queries()
    for query in QueryHolder.queries:
        print(query)


