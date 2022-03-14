import nltk
from nltk.stem.snowball import EnglishStemmer
from argparse import ArgumentParser
import pickle
import os
from bs4 import BeautifulSoup
from tqdm import tqdm
import json
from typing import List
import math


class Indexer:
    """NLTK-based inverted indexer"""

    def __init__(self, use_stemmer=False):
        self.tokenizer = nltk.word_tokenize  # sentence tokenization models
        self.stemmer = None
        if use_stemmer:
            self.stemmer = EnglishStemmer()  # slightly better than Porter stemmer

        # Map from document ID to its URL
        self.doc_id_url_map = {}
        self.doc_id_disk_loc = {}

        # Map between terms and its posting.
        # Posting is currently a tuple (doc_id, term_freq)
        self.inverted_index = {}

        self.__current_id = 0

    # If all URL are distinct we don't even need url_doc_id_map. Leaving it here for now
    def index_document(self, document, url, disk_location):
        # Process docID mapping
        doc_id = self.__current_id
        # Doc ID mapping
        self.doc_id_url_map[doc_id] = url
        self.doc_id_disk_loc[doc_id] = disk_location
        self.__current_id += 1

        for token in self.tokenizer(document):
            token = self.process_token(token)
            if not token:
                continue
            if token not in self.inverted_index:
                self.inverted_index[token] = []
            # If current term is not calculated before -> add to list
            # self.inverted_index[token][-1] is last doc id for current term
            if (
                len(self.inverted_index[token]) == 0
                or self.inverted_index[token][-1][0] != doc_id
            ):
                self.inverted_index[token].append((doc_id, 1))
            else:
                self.inverted_index[token][-1] = (
                    doc_id,
                    self.inverted_index[token][-1][1] + 1,
                )

    # For consistency between inverted index processing and query processing
    def process_token(self, token: str):
        if len(token) == 1 and not token.isalnum():  # Removing separator
            return None
        token = token.lower()
        if self.stemmer:
            token = self.stemmer.stem(token)
        # if token in stopwords: # uncomment to exclude stop words
        #     return None
        return token

    # Query processing
    def process_query(self, query: str):
        processed_query = []
        for token in self.tokenizer(query):
            token = self.process_token(token)
            if token:
                processed_query.append(token)
        return processed_query

    # boolean retrieval model:
    def boolean_retrieval(self, processed_query: List[str]):
        doc_id_lst = []
        term_doc_id_lst = []
        for token in processed_query:
            # TODO: What if token is not in our index?
            # term_doc_id_lst.append([item[0] for item in self.inverted_index[token]])
            term_doc_id_lst.append(self.inverted_index[token])

        # If memory is an issue can move the whole part below to process in the for loop above
        # sort query based on length
        term_doc_id_lst = sorted(term_doc_id_lst, key=lambda x: len(x))
        pointer_lst = [0 for _ in range(len(term_doc_id_lst))]  # list for skip pointer
        N = len(self.doc_id_url_map)
        # Process term with the lowest amount of doc id
        for current_doc_id, term_freq in term_doc_id_lst[0]:
            same_doc_id = True
            for term_idx in range(1, len(term_doc_id_lst)):
                # Two condition: Pointer to doc doesn't go out of bound, and other doc id < current doc id
                # Also setting this to -1 in case the first condition fails
                other_doc_id = -1
                while (
                    pointer_lst[term_idx] < len(term_doc_id_lst[term_idx])
                    and (
                        other_doc_id := term_doc_id_lst[term_idx][
                            pointer_lst[term_idx]
                        ][0]
                    )
                    < current_doc_id
                ):
                    pointer_lst[term_idx] += 1
                if (
                    other_doc_id != current_doc_id
                ):  # If other term doesn't include the current doc id we can skip this doc
                    same_doc_id = False
                    break
            if same_doc_id:
                # term frequency, inverse document frequency
                tfidf = 0
                for token_idx in range(len(term_doc_id_lst)):
                    token_tf = 1 + math.log10(
                        term_doc_id_lst[token_idx][pointer_lst[token_idx]][1]
                    )
                    token_idf = math.log10(N / len(term_doc_id_lst[token_idx]))
                    tfidf += token_tf * token_idf
                # for term_idx, doc_ptr in enumerate(pointer_lst):
                #     total_freq += (1 / len(processed_query)) * term_doc_id_lst[term_idx][pointer_lst[term_idx]][1]
                doc_id_lst.append((current_doc_id, tfidf))
            pointer_lst[0] += 1  # increment pointer for first list also
        doc_id_lst = sorted(doc_id_lst, key=lambda x: x[1], reverse=True)
        doc_id_lst = [x[0] for x in doc_id_lst]
        return doc_id_lst

    def retrieve(self, query, top_k=5):
        processed_query = self.process_query(query)
        print(processed_query)
        doc_ids = self.boolean_retrieval(processed_query)
        # results = sorted(results, key=lambda x: x[1], reverse=True)
        doc_id_results = [self.doc_id_url_map[doc_id] for doc_id in doc_ids]
        disk_loc_results = [self.doc_id_disk_loc[doc_id] for doc_id in doc_ids]
        return doc_id_results[:top_k], disk_loc_results[:top_k]

    def dump_inverted_index(self, path_to_dump):
        with open(path_to_dump, "wb") as f_out:
            pickle.dump(self.inverted_index, f_out, pickle.HIGHEST_PROTOCOL)

    def dump_doc_id_map(self, path_to_dump):
        doc_id_state = {
            "url_map": self.doc_id_url_map,
            "disk_loc": self.doc_id_disk_loc
        }
        with open(path_to_dump, "wb") as f_out:
            pickle.dump(doc_id_state, f_out, pickle.HIGHEST_PROTOCOL)

    def load_inverted_index(self, path_to_load):
        with open(path_to_load, "rb") as f_in:
            self.inverted_index = pickle.load(f_in)

    def load_doc_id_map(self, path_to_load):
        with open(path_to_load, "rb") as f_in:
            doc_id_state = pickle.load(f_in)
        self.doc_id_url_map = doc_id_state["url_map"]
        self.doc_id_disk_loc = doc_id_state["disk_loc"]


def create_indexer(data_path, use_stemmer=False):
    assert os.path.exists(data_path), "Input path does not exist"

    indexer = Indexer(use_stemmer)  # For now let's not use stemmer
    for directory in tqdm(os.listdir(data_path), desc="Whole dataset progress"):
        for file in tqdm(
            os.listdir(os.path.join(data_path, directory)),
            leave=False,
            desc=f"Processing {directory}",
        ):
            file_path = os.path.join(data_path, directory, file)
            with open(file_path) as f:
                content = json.load(f)

            soup = BeautifulSoup(content["content"], "lxml")
            # TODO: Right now all text have equal weight. Need to change importance for title, h1, h2, etc.
            indexer.index_document(soup.text, content["url"], os.path.abspath(file_path))

    return indexer


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--data", type=str, default="data/ANALYST", help="Path to web data"
    )
    args = parser.parse_args()

    index_db = create_indexer(args.data)
    # Hardcode name for now
    print("Saving doc_id map")
    index_db.dump_doc_id_map("doc_id.pkl")
    print("Saving inverted index")
    index_db.dump_inverted_index("inverted_index.pkl")
