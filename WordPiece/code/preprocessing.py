"""Simple Unicode word/punctuation splitting, shared by training and encoding."""

import re


def pre_tokenize(text, lowercase=False):
    if lowercase:
        text = text.lower()
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
