from indexer import Indexer
from utils import parse_arguments
import time

args = parse_arguments()

# Load indexer
indexer = Indexer()
indexer.load_inverted_index(args.inverted_index_file)
indexer.load_doc_id_map(args.doc_id_file)

# Retrieve
def main():
    queries = [
        "cristina lopes",
        "machine learning",
        "ACM",
        "master of software engineering",
    ]

    for query in queries:
        start = time.time()
        results = indexer.retrieve(query, top_k=5)
        for result in results:
            print(result)

        print(f"Retrieval took {(time.time() - start):.3f}s")
        print()


if __name__ == "__main__":
    main()
