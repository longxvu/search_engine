import glob
import os
import json
import pickle

DATA_PATH = "./dev_data.pkl"


def read_from_dir():
    if not os.path.exists(DATA_PATH):
        sub_domains = glob.glob("DEV/*")
        url_content_dict = {}
        num = 0
        for sub_domain in sub_domains:
            crawled_file = sub_domain + "/*"
            crawled_files = sub_domains = glob.glob(crawled_file)
            for f in crawled_files:
                num += 1
                with open(f, "rb") as read_file:
                    data = json.load(read_file)
                url = data["url"]
                content = data["content"]
                encoding = data["encoding"]  # seems to be not useful
                url_content_dict[url] = content
        print(f"loaded {num} files")
        dump_data(url_content_dict, DATA_PATH)
    else:
        url_content_dict = load_data(DATA_PATH)
        print(f"loaded {len(url_content_dict)} files from saved state")
    return url_content_dict


def dump_data(data, path_to_dump):  # save the data so that indexing can be faster
    with open(path_to_dump, "wb") as f_out:
        pickle.dump(data, f_out, pickle.HIGHEST_PROTOCOL)


def load_data(path_to_load):
    with open(path_to_load, "rb") as f_in:
        data = pickle.load(f_in)
    return data
