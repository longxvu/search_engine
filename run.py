from indexer import Indexer
from utils import parse_arguments
import time


arguments = parse_arguments()

def run(args):
    indexer = Indexer()
    indexer.load_inverted_index(args.inverted_index_file)
    indexer.load_doc_id_map(args.doc_id_file)

    while True:
        input_query = input("Enter your search query: ")
        start = time.time()
        results = indexer.retrieve(input_query, top_k=5)
        end = time.time()
        print(f'Results for "{input_query}":')
        for result in results:
            print(result)
        print(f"Retrieval took {(end - start):.3f}s")
        print()

if __name__ == "__main__":
    run(arguments)
