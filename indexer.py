import re

import nltk
from nltk.stem.snowball import EnglishStemmer
import pickle
import os
from bs4 import BeautifulSoup
from tqdm import tqdm
import json
from typing import List
import math
from posting import Posting
import shutil
from utils import parse_config
from urllib.parse import urldefrag
import time


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

        # Map from term to its location in final posting file
        self.term_posting_map = {}

        # Map from term to its file, used for partial index
        self.__term_file_partial_map = {}
        self.__current_partial_index_file_id = 0

        # Set of discovered URL, maybe useful for duplication, now contains URL without fragment
        self.__discovered_url = set()

        # all term posting location
        self.term_posting_path = None

        # Map between terms and its posting.
        # Posting is currently
        self.inverted_index = {}
        self.inverted_bigram_index = {}

        self.__current_id = 0
        self.__current_approximate_size = 0

        self.__non_alpha_numeric_pattern = re.compile("^[^a-zA-Z\d]+$")

    def index_document(self, document, url, disk_location, temp_dir, partial_max_size):
        # Defragment as soft duplication detection
        defragmented_url, _ = urldefrag(url)
        if defragmented_url in self.__discovered_url:  # skip if already exists
            return
        self.__discovered_url.add(defragmented_url)

        # Process docID mapping
        doc_id = self.__current_id
        # Doc ID mapping
        self.doc_id_url_map[doc_id] = defragmented_url
        self.doc_id_disk_loc[doc_id] = disk_location
        self.__current_id += 1

        # posting for this doc
        doc_posting_dict = {}

        token_pos = 0
        token_list = [self.process_token(token) for token in self.tokenizer(document)]
        token_list = [token for token in token_list if token is not None]
        for token in token_list:
            if token not in doc_posting_dict:
                doc_posting_dict[token] = Posting()

            doc_posting_dict[token].doc_id = doc_id
            doc_posting_dict[token].update_position_list(token_pos)
            token_pos += 1

        self.update_inverted_index(doc_posting_dict, temp_dir, partial_max_size)

        # bigram construction
        for bigram_token in zip(token_list[:-1], token_list[1:]):
            bigram_posting = Posting()
            bigram_posting.doc_id = doc_id
            if bigram_token not in self.inverted_bigram_index:
                self.inverted_bigram_index[bigram_token] = []
            else:
                if self.inverted_bigram_index[bigram_token][-1].doc_id == doc_id:
                    bigram_posting = self.inverted_bigram_index[bigram_token].pop()  # Same document just retrieved last post

            bigram_posting.term_freq += 1
            self.inverted_bigram_index[bigram_token].append(bigram_posting)


    # update posting for current document into list of inverted_index posting
    # guarantee posting is sorted
    def update_inverted_index(self, doc_posting, temp_dir, partial_max_size):
        for term, posting in doc_posting.items():
            if term not in self.inverted_index:
                self.inverted_index[term] = []
            self.inverted_index[term].append(posting)
            self.__current_approximate_size += posting.get_approximate_size()

        if self.__current_approximate_size > partial_max_size:
            self.save_partial_index(temp_dir)

    def save_partial_index(self, temp_dir):
        if len(self.inverted_index) == 0:
            return
        os.makedirs(temp_dir, exist_ok=True)
        tmp_partial_path = os.path.join(temp_dir, f"partial_tmp_{self.__current_partial_index_file_id:06}")
        with open(tmp_partial_path, "w") as f:
            for term, postings in self.inverted_index.items():
                posting_start = f.tell()
                for posting in postings:
                    f.write(str(posting) + "\n")
                posting_length = f.tell() - posting_start
                if term not in self.__term_file_partial_map:
                    self.__term_file_partial_map[term] = []

                self.__term_file_partial_map[term].append((self.__current_partial_index_file_id,
                                                           posting_start,
                                                           posting_length))
        self.__current_partial_index_file_id += 1
        self.inverted_index = {}                # Reset inverted index
        self.__current_approximate_size = 0     # Reset processing size

    # Merge partial index
    def merge_and_write_partial_posting(self, path_to_dump, temp_dir):
        # Writing any posting left in inverted index
        self.save_partial_index(temp_dir)

        tqdm.write("Merging partial index")
        # Open final posting file to write
        with open(path_to_dump, "w") as f_out:
            for term, postings in tqdm(self.__term_file_partial_map.items()):
                final_term_posting_list = []
                # Get partial posting from multiple files
                for file_idx, posting_start, posting_length in postings:
                    partial_file_path = os.path.join(temp_dir, f"partial_tmp_{file_idx:06}")
                    # f.seek problems with new line on Windows, so we have to read it in binary mode and decode the str
                    with open(partial_file_path, "rb") as f_in:
                        f_in.seek(posting_start)
                        content = f_in.read(posting_length)
                        final_term_posting_list.extend(parse_multiple_posting(content.decode("utf-8")))

                 # Write to final file. Mapping term to its posting position in final merged file
                posting_start = f_out.tell()
                for posting in final_term_posting_list:
                    f_out.write(str(posting) + "\n")
                posting_length = f_out.tell() - posting_start
                self.term_posting_map[term] = (posting_start, posting_length)

    # For consistency between inverted index processing and query processing
    def process_token(self, token: str):
        if self.__non_alpha_numeric_pattern.match(token):  # Removing non-alphanumeric token
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
        doc_id_score_map = {}
        term_doc_id_lst = []
        bigram_lst = []
        for token in processed_query:
            start = time.time()
            posting_found = self.load_posting_from_disk(token)
            print(f"Retrieving '{token}': {time.time() - start:.3f}s")
            if posting_found:
                term_doc_id_lst.append(posting_found)

        if len(processed_query) >= 2:
            for bigram_token in zip(processed_query[:-1], processed_query[1:]):
                bigram_lst.append(bigram_token)
        # No results found for given query
        if len(term_doc_id_lst) == 0:
            return []
        # sort query based on length of its posting list
        print(bigram_lst)
        term_doc_id_lst = sorted(term_doc_id_lst, key=lambda x: len(x))
        pointer_lst = [0 for _ in range(len(term_doc_id_lst))]  # list for skip pointer
        N = len(self.doc_id_url_map)
        # Process term with the lowest amount of doc id
        for posting in term_doc_id_lst[0]:
            current_doc_id = posting.doc_id
            same_doc_id = True

            for term_idx in range(1, len(term_doc_id_lst)):
                # Two condition: Pointer to doc doesn't go out of bound, and other doc id < current doc id
                # Also setting this to -1 in case the first condition fails
                other_doc_id = -1
                while (pointer_lst[term_idx] < len(term_doc_id_lst[term_idx])
                    and (other_doc_id := term_doc_id_lst[term_idx][pointer_lst[term_idx]].doc_id) < current_doc_id):
                    pointer_lst[term_idx] += 1
                # If other term doesn't include the current doc id we can skip this doc
                if other_doc_id != current_doc_id:
                    same_doc_id = False
                    break
            if same_doc_id:
                # term frequency, inverse document frequency
                tfidf = 0
                for token_idx in range(len(term_doc_id_lst)):
                    token_tf = 1 + math.log10(
                        term_doc_id_lst[token_idx][pointer_lst[token_idx]].term_freq
                    )
                    token_idf = math.log10(N / len(term_doc_id_lst[token_idx]))
                    tfidf += token_tf * token_idf
                # for term_idx, doc_ptr in enumerate(pointer_lst):
                #     total_freq += (1 / len(processed_query)) * term_doc_id_lst[term_idx][pointer_lst[term_idx]][1]
                doc_id_score_map[current_doc_id] = [tfidf, 0]  # First idx is tfidf for unigram, second idx is for bigram
            pointer_lst[0] += 1  # increment pointer for first list also

        # doc_id_lst = sorted(doc_id_lst, key=lambda x: x[1], reverse=True)
        # doc_id_results = [x[0] for x in doc_id_lst]

        # computing tf-idf score for bigram for the given doc id list
        if len(bigram_lst) > 0:
            for bigram_token in bigram_lst:
                if bigram_token in self.inverted_bigram_index:
                    doc_containing_bigram = []
                    for posting in self.inverted_bigram_index[bigram_token]:
                        if posting.doc_id in doc_id_score_map:
                            doc_containing_bigram.append(posting)
                    # doc_containing_bigram = [posting for posting in self.inverted_bigram_index[bigram_token]
                    #                          if posting.doc_id in doc_id_score_map]
                    for posting in doc_containing_bigram:
                        bigram_tf = 1 + math.log10(posting.term_freq)
                        bigram_idf = math.log10(N / len(self.inverted_bigram_index[bigram_token]))
                        bigram_tfidf = bigram_tf * bigram_idf
                        doc_id_score_map[posting.doc_id][1] = bigram_tfidf

        # compute final score for each doc id
        unigram_weight = 0.3
        bigram_weight = 0.7
        doc_id_score_map = {doc_id: unigram_weight * scores[0] + bigram_weight * scores[1]
                            for doc_id, scores in doc_id_score_map.items()}
        doc_id_results = [k for k, v in sorted(doc_id_score_map.items(), key=lambda x: x[1], reverse=True)]

        return doc_id_results

    def retrieve(self, query, top_k=5):
        processed_query = self.process_query(query)
        print(processed_query)
        doc_ids = self.boolean_retrieval(processed_query)
        # results = sorted(results, key=lambda x: x[1], reverse=True)
        doc_id_results = [self.doc_id_url_map[doc_id] for doc_id in doc_ids]
        disk_loc_results = [self.doc_id_disk_loc[doc_id] for doc_id in doc_ids]
        return doc_id_results[:top_k], disk_loc_results[:top_k]

    def dump_indexer_state(self, dir_to_dump, doc_id_file, all_posting_file, term_posting_file, bigram_file, temp_dir):
        os.makedirs(dir_to_dump, exist_ok=True)

        print("Saving doc_id map")
        self.__dump_doc_id_map(os.path.join(dir_to_dump, doc_id_file))
        print("Merging and writing partial posting")
        self.merge_and_write_partial_posting(os.path.join(dir_to_dump, all_posting_file), temp_dir)
        print("Saving bigram index")
        self.__dump_bigram_index(os.path.join(dir_to_dump, bigram_file))
        print("Saving term posting map")
        self.__dump_term_posting_map(os.path.join(dir_to_dump, term_posting_file))  # Save this last
        print(f"Done. Indexer state dumped to: {dir_to_dump}")


    def load_indexer_state(self, dir_to_load, doc_id_file, all_posting_file, term_posting_file, bigram_file):
        print("Loading doc_id map")
        self.__load_doc_id_map(os.path.join(dir_to_load, doc_id_file))
        print("Loading term posting map")
        self.__load_term_posting_map(os.path.join(dir_to_load, term_posting_file))
        print("Loading bigram index")
        self.__load_bigram_index(os.path.join(dir_to_load, bigram_file))

        self.term_posting_path = os.path.join(dir_to_load, all_posting_file)
        print("Indexer state loaded")

    def load_posting_from_disk(self, term):
        if term not in self.term_posting_map:
            return None
        posting_start, posting_length = self.term_posting_map[term]
        start = time.time()
        with open(self.term_posting_path, "rb") as f:
            f.seek(posting_start)
            content = f.read(posting_length)
            mid = time.time()
        postings = parse_multiple_posting(content.decode("utf-8"))
        end = time.time()
        print(f"Reading took {mid - start:.3f}s")
        print(f"Parsing took {end - mid:.3f}s")
        return postings


    def __dump_term_posting_map(self, path_to_dump):
        with open(path_to_dump, "wb") as f_out:
            pickle.dump(self.term_posting_map, f_out, pickle.HIGHEST_PROTOCOL)

    def __dump_doc_id_map(self, path_to_dump):
        doc_id_state = {
            "url_map": self.doc_id_url_map,
            "disk_loc": self.doc_id_disk_loc
        }
        with open(path_to_dump, "wb") as f_out:
            pickle.dump(doc_id_state, f_out, pickle.HIGHEST_PROTOCOL)

    def __dump_bigram_index(self, path_to_dump):
        with open(path_to_dump, "wb") as f_out:
            pickle.dump(self.inverted_bigram_index, f_out, pickle.HIGHEST_PROTOCOL)

    def __load_term_posting_map(self, path_to_load):
        with open(path_to_load, "rb") as f_in:
            self.term_posting_map = pickle.load(f_in)

    def __load_doc_id_map(self, path_to_load):
        with open(path_to_load, "rb") as f_in:
            doc_id_state = pickle.load(f_in)
        self.doc_id_url_map = doc_id_state["url_map"]
        self.doc_id_disk_loc = doc_id_state["disk_loc"]

    def __load_bigram_index(self, path_to_load):
        with open(path_to_load, "rb") as f_in:
            self.inverted_bigram_index = pickle.load(f_in)


def create_indexer(data_path, temp_dir, partial_max_size, use_stemmer=False):
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
            indexer.index_document(soup.text, content["url"], os.path.abspath(file_path), temp_dir, partial_max_size)

    return indexer

# TODO: Can introduce max load time here to prevent time parsing common word
def parse_multiple_posting(posting_str: str):
    result = []
    posting_str = posting_str.splitlines()
    for i in range(0, len(posting_str), 2):
        posting = Posting()
        posting.parse_from_str(posting_str[i:i + 2])
        result.append(posting)
    return result


if __name__ == "__main__":
    default_config, data_config = parse_config()

    tmp_dir = default_config["tmp_dir"]
    # Delete partial index before creating indexer
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

    index_db = create_indexer(data_config["data_path"],
                              tmp_dir,
                              int(float(data_config["partial_max_size"])))
    # Hardcode name for now
    index_db.dump_indexer_state(data_config["indexer_state_dir"],
                                default_config["doc_id_file"],
                                default_config["all_posting_file"],
                                default_config["term_posting_map_file"],
                                default_config["bigram_file"],
                                tmp_dir)
