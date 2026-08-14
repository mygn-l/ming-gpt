import random

from vocabularizer import UNK_TOKEN


def build_trie(sorted_vocabulary):
    root = {}
    for word in sorted_vocabulary:
        node = root
        for char in word:
            if char not in node:
                node[char] = {}
            node = node[char]
        node[""] = word
    return root


def tokenize(sorted_vocabulary, text):
    text = text.lower()
    root = build_trie(sorted_vocabulary)

    tokens = []
    i = 0
    n = len(text)
    while i < n:
        node = root
        j = i
        last = None
        while j < n and text[j] in node:
            node = node[text[j]]
            j += 1
            if "" in node:
                last = (j, node[""])
        if last is not None:
            tokens.append(last[1])
            i = last[0]
        else:
            tokens.append(UNK_TOKEN)
            i += 1

        if random.random() > 0.999999:
            print(f"Progress: {i}/{n}")
    return tokens
