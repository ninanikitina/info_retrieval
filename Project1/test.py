term_freq = {
    "a": {1: 3, 2: 5, 3: 6},
    "b": {1: 3, 2: 5, 3: 6},
    "c": {1: 3, 2: 5, 7: 6},
    "d": {1: 3, 44: 5, 8: 6},
    "e": {1: 3, 8: 5, 6: 6},
    "f": {1: 3, 7: 5, 9: 6}
}
query = ['a', 'b', 'c', 'd']

doc_w_least_one_term = set()
all_docs = set()
docs_term_sets = []
for i, w in enumerate(query):
    if w in term_freq:
        docs = set(k for k, v in term_freq[w].items())
        temp = docs.copy()
        docs_term_sets.append(temp)
        if i == 0:
            all_docs = docs
        else:
            all_docs.intersection_update(docs)
        doc_w_least_one_term.update(docs)
#  If there are less than 50 documents, then consider resources in DC
#  that contain n-1 terms in q (all the combinations), then n-2 ect.
min_q = len(query) - 1
while len(all_docs) < 50 and min_q >= 1:
    doc_w_least_one_term.difference_update(all_docs)
    for d in doc_w_least_one_term:
        count = 0
        for s in docs_term_sets:
            if d in s:
                count = count + 1
        if count >= min_q:
            all_docs.add(d)
    min_q = min_q - 1
print(all_docs)
