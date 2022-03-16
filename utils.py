from argparse import ArgumentParser
from posixpath import join as urljoin
from bs4 import BeautifulSoup
import configparser
import json
import os
import re
import sys   
import nltk
import hashlib
import numpy as np
sys.setrecursionlimit(10000)

def parse_config(config_file="config/config.ini"):
    parser = ArgumentParser()
    parser.add_argument(
        "--config", type=str, choices=["ANALYST", "DEV"], default="ANALYST", help="Path to web data"
    )
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(config_file)
    default_config = config["DEFAULT"]
    data_config = config[args.config]

    return default_config, data_config


def generate_result_pages(disk_locs, static_dir, generated_result_dir, query=None):
    generated_pages = []
    result_dir = os.path.join(static_dir, generated_result_dir)
    os.makedirs(result_dir, exist_ok=True)

    for idx, disk_loc in enumerate(disk_locs):
        with open(disk_loc, encoding="utf8") as f:
            content = json.load(f)
        content = content["content"]
        # highlighting query if provided
        if query:
            content = highlight_html(content, query)

        file_name = f"result_{idx:02}.html"
        path = os.path.join(result_dir, file_name)
        with open(path, "w", encoding="utf8") as f:
            f.write(content)

        generated_pages.append(urljoin(generated_result_dir, file_name))

    return generated_pages


def highlight_html(html_str, query):
    soup = BeautifulSoup(html_str, "html.parser")
    query = query.split()
    query = list(set(query))

    # Pattern for complete match for word, not partial match
    patterns = [re.compile(f"(^|[^a-zA-Z\d]+|\s+)({word})($|[^a-zA-Z\d]+|\s+)", flags=re.IGNORECASE) for word in query]
    for pattern in patterns:
        for tag in soup.find_all(text=pattern):
            highlighted = re.sub(pattern, "\g<1><mark>\g<2></mark>\g<3>", tag)
            tag.replace_with(BeautifulSoup(highlighted, "html.parser"))

    return str(soup)

    def find_weights(content):
    weights = {}
    for token in content:
        if token in weights:
            weights[token] += 1
        else:
            weights[token] = 1
            
    return weights

def hash_function(token, num_bits=64):
    return hashlib.md5(token.encode('utf-8')).digest()[int(-num_bits/8):]

def compute_similarity(val, another_val, num_bits=64):    
    count = 0
    
    for i in range(0,num_bits):
        if ((( val>> i) & 1) == (( another_val>>i) & 1)):
            count += 1
            
    return count/num_bits

def gen_hash(weights, num_bits=64):
    hash_values = []
    v = [0] * num_bits
    weights = weights.items()
    for token,weight in weights:
        h = hash_function(token)
        bitarray = np.unpackbits(np.frombuffer(h, dtype=np.uint8))
        for i in range(num_bits):
            if bitarray[i] == 1:
                v[i] += weight
            else:
                v[i] -= weight
                
    for i in range(num_bits):
        if v[i] > 0:
            v[i] = 1
        else:
            v[i] = 0

    int_hash_val = int.from_bytes(np.packbits(v).tobytes(), 'big')
    return int_hash_val