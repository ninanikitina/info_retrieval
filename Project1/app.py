from flask import Flask
from flask import render_template

app = Flask(__name__)


@app.route('/')
def hello():
    return render_template('home.html')

@app.route('/suggestions', methods=["POST"])
def get_suggestions():
    pass

@app.route('/search', methods=["POST"])
def get_search_articles():
    pass

