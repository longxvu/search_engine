from indexer import Indexer
from utils import parse_config
import time

def run():
    default_config, data_config = parse_config()
    indexer = Indexer()
    indexer.load_indexer_state(data_config["indexer_state_dir"],
                               default_config["doc_id_file"],
                               default_config["all_posting_file"],
                               default_config["term_posting_map_file"],
                               default_config["bigram_file"],
                               default_config["bigram_partial_file"],
                               default_config["bigram_partial_file_map"])

    while True:
        input_query = input("Enter your search query: ")
        start = time.time()
        results = indexer.retrieve(input_query, top_k=int(default_config["max_result"]))
        end = time.time()
        print(f'Results for "{input_query}":')
        for result in results:
            print(result)
        print(f"Retrieval took {(end - start):.3f}s")
        print()

if __name__ == "__main__":
    run()
