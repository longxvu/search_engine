import re
from collections import Counter
import math

word_pattern = re.compile("[a-zA-Z0-9@#*&']+")


def tokenize_word(input_str, include_separator=False):
    return tokenize(input_str, word_pattern, include_separator)


def tokenize(input_str, compiled_pattern, include_separator):
    token_list = []
    last_found = 0

    for match in compiled_pattern.finditer(input_str):
        # get matched token start and end index
        token_start, token_end = match.span()

        # add separator (except space and non-English characters) to token list if there exists any
        if include_separator:
            if token_start != last_found:
                token_list.extend(
                    [
                        c
                        for c in input_str[last_found:token_start]
                        if c != " " and c.isascii()
                    ]
                )
        # add matched token to token list
        token_list.append(input_str[token_start:token_end].lower())
        last_found = token_end

    # process any separator at the end of the string
    if include_separator:
        if last_found != len(input_str):
            token_list.extend(
                [
                    c
                    for c in input_str[last_found : len(input_str)]
                    if c != " " and c.isascii()
                ]
            )

    return token_list


def compute_cosine_similarity(token_list_1, token_list_2):
    l1_counter = Counter(token_list_1)
    l2_counter = Counter(token_list_2)

    intersection = set(l1_counter.keys()) & set(l2_counter.keys())
    numerator = sum(
        l1_counter[token] * l2_counter[token] for token in intersection
    )  # A dot B

    l1_mag = math.sqrt(sum(l1_counter[token] ** 2 for token in l1_counter))
    l2_mag = math.sqrt(sum(l2_counter[token] ** 2 for token in l2_counter))
    denominator = l1_mag * l2_mag  # ||A||*||B||

    if denominator == 0:
        return 0
    else:
        return numerator / denominator
