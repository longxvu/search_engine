import flask
from indexer import Indexer
import time
from flask import Flask, render_template, request
from utils import parse_arguments

app = flask.Flask(__name__)
app.config["DEBUG"] = True

args = parse_arguments()
# Load indexer
indexer = Indexer()
indexer.load_inverted_index(args.inverted_index_file)
indexer.load_doc_id_map(args.doc_id_file)


@app.route("/", methods=["GET"])
def home():
    return render_template("search_engine.html")


@app.route("/search", methods=["POST"])
def search():
    query = request.form.get("input")
    print(query)
    start = time.time()
    results = indexer.retrieve(query, top_k=5)
    t = time.time() - start
    ts = f"Retrieval took {(t):.3f}s"
    print(results)
    print(ts)
    results.append(ts)
    return render_template("search_engine.html", results=results)


app.run()
