import flask
from argparse import ArgumentParser
from indexer import Indexer
import time
from flask import Flask, render_template, request
from m2_report import indexer

app = flask.Flask(__name__)
app.config["DEBUG"] = True


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
