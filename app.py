import flask
from indexer import Indexer
import time
from flask import render_template, request
from utils import parse_arguments, generate_result_pages

app = flask.Flask(__name__,
                  static_folder="web/static",
                  template_folder="web/templates")
app.config["DEBUG"] = True

args = parse_arguments()
# Load indexer
indexer = Indexer()
indexer.load_inverted_index(args.inverted_index_file)
indexer.load_doc_id_map(args.doc_id_file)


@app.route("/", methods=["GET"])
def home():
    return render_template("search_engine.html", data=None)


@app.route("/search", methods=["POST"])
def search():
    query = request.form.get("input")
    print(query)
    start = time.time()
    url_map, disk_locs = indexer.retrieve(query, top_k=5)
    ts = f"Retrieval took {(time.time() - start):.3f}s"

    generated_results = generate_result_pages(disk_locs, query)
    print(generated_results)
    data = {
        "results": list(zip(url_map, generated_results)),
        "ts": ts
    }
    print(data)
    return render_template("search_engine.html", data=data)


app.run()
