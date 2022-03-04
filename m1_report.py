import argparse
import time
import os
from indexer import Indexer

parser = argparse.ArgumentParser()
parser.add_argument("--inverted_index_file", default="inverted_index.pkl")
parser.add_argument("--doc_id_file", default="doc_id.pkl")

args = parser.parse_args()

start = time.time()
indexer = Indexer()
indexer.load_inverted_index(args.inverted_index_file)
indexer.load_doc_id_map(args.doc_id_file)
print(f"Indexer loaded in {(time.time() - start):.3f}s")

print(f"Number of indexed documents: {len(indexer.doc_id_url_map)}")
print(f"Number of unique tokens: {len(indexer.inverted_index)}")
print(f"Inverted index file size: {(os.path.getsize(args.inverted_index_file) / 1024):.4f} KB")
print(f"Doc ID file size: {(os.path.getsize(args.doc_id_file) / 1024):.4f} KB")
