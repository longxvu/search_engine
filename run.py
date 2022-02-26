from argparse import ArgumentParser
import json
import os
from bs4 import BeautifulSoup
from indexer import Indexer, create_indexer
from tqdm import tqdm


parser = ArgumentParser()
parser.add_argument("--data", type=str, default="data/ANALYST", help="Path to web data")

arguments = parser.parse_args()

def run(args):
    # Create indexer in memory
    indexer = create_indexer(args.data)
    # Hardcode name for now
    print("Saving doc_id map")
    indexer.dump_doc_id_map("doc_id.pkl")
    print("Saving inverted index")
    indexer.dump_inverted_index("inverted_index.pkl")

if __name__ == "__main__":
    run(arguments)
