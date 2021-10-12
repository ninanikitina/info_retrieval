from flask import Flask
from flask import render_template
from flask import request
from flask_cors import CORS
from waitress import serve
import QueryLogger
app = Flask(__name__)
CORS(app)

queries = QueryLogger.QueryLogger()

@app.route('/')
def hello():
    return render_template('home.html')

@app.route('/suggestions', methods=["POST"])
def get_suggestions():
    print(request.json["query"])
    return queries.get_suggestions(request.json["query"])

@app.route('/search', methods=["POST"])
def get_search_articles():
    pass

if __name__ == "__main__":
  print("Serving hosting via Waitress on port 5000")
  serve(app, listen='*:5000')

