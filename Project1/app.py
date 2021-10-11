from flask import Flask
from flask import render_template
from flask import request
from flask_cors import CORS
import QueryLogger

app = Flask(__name__)
CORS(app)
queries = QueryLogger.QueryLogger()

@app.route('/')
def hello():
    return render_template('home.html')

@app.route('/suggestions', methods=["POST"])
def get_suggestions():
    return queries.get_suggestions(request.form["query"])

@app.route('/search', methods=["POST"])
def get_search_articles():
    pass

