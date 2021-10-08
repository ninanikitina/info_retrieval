class QueryHolder(object):
    def __init__(self, QueryObj):
        self.queries = set()

    def find_query(self, id):
        for x in self.queries: # Search queries for one that starts with our position
            if x.id == id:
                return x

    def add_query(self, QueryObj): # Adds QueryObj
        self.queries.add(QueryObj)




