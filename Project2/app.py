from flask import Flask
from flask import render_template
from flask import request
from flask_cors import CORS
from waitress import serve
app = Flask(__name__)
CORS(app)
import asyncio
from main import SearchEngineData as SED
from QueryLogger import QueryLogger

x = QueryLogger()
data = SED()

@app.route('/')
def hello():
    x.load_queries()
    return render_template('home.html')

@app.route('/suggestions', methods=["POST"])
def get_suggestions():
    return x.get_suggestions(request.json["query"])

@app.route('/search', methods=["POST"])
def get_search_articles():
    print(f"Searching documents in regard to: {request.json['query']}")
    print(f"Be patient as we search for your documents.")
    asyncio.create_task(x.add_query(request.json["query"]))
    asyncio.create_task(x.save_recommendations())
    return data.run_query(request.json["query"])

if __name__ == "__main__":
  print("Hosting via Waitress on localhost:5000")
  serve(app, listen='*:5000')

