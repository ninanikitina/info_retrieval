from flask import Flask
from flask import render_template
from flask import request
from flask_cors import CORS
from waitress import serve
import QueryLogger
import main
app = Flask(__name__)
CORS(app)

data = main.SearchEngineData()
data.query_logger.load_queries()


@app.route('/')
def hello():
    return render_template('home.html')

@app.route('/suggestions', methods=["POST"])
def get_suggestions():
    return data.query_logger.get_suggestions(request.json["query"])

@app.route('/add_query', methods=["POST"])
def add_query():
    data.query_logger.add_query(request.json["query"])

@app.route('/search', methods=["POST"])
def get_search_articles():
    return data.run_query(request.json["query"])

if __name__ == "__main__":
  print("Hosting via Waitress on localhost:5000")
  serve(app, listen='*:5000')

