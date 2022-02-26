import nltk
from nltk.stem.snowball import EnglishStemmer
from nltk.corpus import stopwords
import pickle
import os
from bs4 import BeautifulSoup
from tqdm import tqdm
import json


class Indexer:
    """ NLTK-based inverted indexer"""
    def __init__(self, use_stemmer=True, ignore_stopwords=False):
        self.tokenizer = nltk.word_tokenize  # sentence tokenization models
        self.stemmer = None
        self.stopwords = None
        if use_stemmer:
            self.stemmer = EnglishStemmer()  # slightly better than Porter stemmer

        if ignore_stopwords:
            self.stopwords = stopwords.words("english")

        # Map between document ID and its URL
        self.doc_id_url_map = {}
        self.url_doc_id_map = {}

        # Map between terms and its posting.
        # Posting is currently a tuple (doc_id, term_freq)
        self.inverted_index = {}

        self.__current_id = 0

    # If all URL are distinct we don't even need url_doc_id_map. Leaving it here for now
    def index_document(self, document, url):
        # Process docID mapping
        if url in self.url_doc_id_map:
            doc_id = self.url_doc_id_map[url]
        else:
            doc_id = self.__current_id
            # Doc ID mapping
            self.doc_id_url_map[doc_id] = url
            self.url_doc_id_map[url] = doc_id
            self.__current_id += 1

        for token in [t.lower() for t in self.tokenizer(document)]:
            if self.stopwords and token in self.stopwords:
                continue
            if len(token) == 1 and not token.isalnum():  # Removing separator
                continue
            if self.stemmer:
                token = self.stemmer.stem(token)

            if token not in self.inverted_index:
                self.inverted_index[token] = []
            # If current term is not calculated before -> add to list
            # self.inverted_index[token][-1] is last doc id for current term
            if len(self.inverted_index[token]) == 0 or self.inverted_index[token][-1][0] != doc_id:
                self.inverted_index[token].append((doc_id, 1))
            else:
                self.inverted_index[token][-1] = (doc_id, self.inverted_index[token][-1][1] + 1)

    def dump_inverted_index(self, path_to_dump):
        with open(path_to_dump, "wb") as f_out:
            pickle.dump(self.inverted_index, f_out, pickle.HIGHEST_PROTOCOL)

    def dump_doc_id_map(self, path_to_dump):
        with open(path_to_dump, "wb") as f_out:
            pickle.dump(self.doc_id_url_map, f_out, pickle.HIGHEST_PROTOCOL)

    def load_inverted_index(self, path_to_load):
        with open(path_to_load, "rb") as f_in:
            self.inverted_index = pickle.load(f_in)

    def load_doc_id_map(self, path_to_load):
        with open(path_to_load, "rb") as f_in:
            self.doc_id_url_map = pickle.load(f_in)


def create_indexer(data_path):
    assert os.path.exists(data_path), "Input path does not exist"

    indexer = Indexer(use_stemmer=False)  # For now let's not use stemmer
    for directory in tqdm(os.listdir(data_path), desc="Whole dataset progress"):
        for file in tqdm(os.listdir(os.path.join(data_path, directory)), leave=False, desc=f"Processing {directory}"):
            file_path = os.path.join(data_path, directory, file)
            with open(file_path) as f:
                content = json.load(f)

            soup = BeautifulSoup(content["content"], "lxml")
            # TODO: Right now all text have equal weight. Need to change importance for title, h1, h2, etc.
            indexer.index_document(soup.text, content["url"])

    return indexer