from argparse import ArgumentParser

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--inverted_index_file", default="inverted_index.pkl")
    parser.add_argument("--doc_id_file", default="doc_id.pkl")
    args = parser.parse_args()

    return args