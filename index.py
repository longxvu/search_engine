from parse_url import read_from_dir
from utils import tokenize_word
from bs4 import BeautifulSoup
from collections import defaultdict, Counter

index = defaultdict(list)
url_mapping = {}
data = read_from_dir()
n = 0
for url in data:
    if url not in url_mapping:
        url_mapping[url] = n
        content = data[url]
        soup = BeautifulSoup(content, "lxml")
        page_tokens = tokenize_word(soup.text)
        freq_dict = Counter(page_tokens)
        for t in freq_dict:
            index[t].append(n)
        n += 1
    if n > 3:
        break
print(index)
# print(url_mapping)
