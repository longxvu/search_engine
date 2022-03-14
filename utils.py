from argparse import ArgumentParser
from posixpath import join as urljoin
from bs4 import BeautifulSoup
import json
import os
import re

WEB_DIR = "web/static"
GENERATED_RESULT_DIR = "generated_results"

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--inverted_index_file", default="inverted_index.pkl")
    parser.add_argument("--doc_id_file", default="doc_id.pkl")
    args = parser.parse_args()

    return args


def generate_result_pages(disk_locs, query=None):
    generated_pages = []
    result_dir = os.path.join(WEB_DIR, GENERATED_RESULT_DIR)
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

        generated_pages.append(urljoin(GENERATED_RESULT_DIR, file_name))

    return generated_pages


def highlight_html(html_str, query):
    soup = BeautifulSoup(html_str, "html.parser")
    query = query.split()
    query = list(set(query))
    patterns = [re.compile(word, flags=re.IGNORECASE) for word in query]

    for pattern in patterns:
        for tag in soup.find_all(text=pattern):
            highlighted = re.sub(pattern, "<mark>\g<0></mark>", tag)
            tag.replace_with(BeautifulSoup(highlighted, "html.parser"))

    return str(soup)