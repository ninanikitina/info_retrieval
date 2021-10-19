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



@app.route('/')
def hello():
    data.query_logger.load_queries()
    return render_template('home.html')

@app.route('/suggestions', methods=["POST"])
def get_suggestions():
    return data.query_logger.get_suggestions(request.json["query"])

@app.route('/search', methods=["POST"])
def get_search_articles():
    print(f"Searching documents in regard to: {request.json['query']}")
    print(f"Be patient as we search for your documents.")
    data.query_logger.add_query(request.json["query"])
    return data.run_query(request.json["query"])

if __name__ == "__main__":
  print("Hosting via Waitress on localhost:5000")
  serve(app, listen='*:5000')

