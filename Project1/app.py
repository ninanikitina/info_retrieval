from flask import Flask
from flask import render_template
from flask import request
import QueryLogger

app = Flask(__name__)

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

if __name__ == "__main__":
  app.run()

