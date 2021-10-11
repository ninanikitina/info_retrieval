from flask import Flask
from flask import render_template
from flask import request
from . import QueryLogger

app = Flask(__name__)
queries = QueryLogger()

@app.route('/')
def hello():
    return render_template('home.html')

@app.route('/suggestions', methods=["POST"])
def get_suggestions():
    data = request.json
    return queries.get_suggestions(data["query"])

@app.route('/search', methods=["POST"])
def get_search_articles():
    pass

