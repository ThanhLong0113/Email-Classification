import re
import numpy as np
import pandas as pd
from gensim.parsing.preprocessing import strip_non_alphanum, split_alphanum, strip_short, strip_numeric
from train import *


def raw_text_preprocess(raw):
    raw = re.sub(r"http\S+", "", raw)  # bỏ đường dẫn
    # loại bỏ kí tự lạ, chuẩn hóa
    raw = strip_non_alphanum(raw).lower().strip()
    raw = split_alphanum(raw)  # tách các từ vô nghĩa
    raw = strip_short(raw, minsize=2)   # loại bỏ kí tự đứng 1 mình
    raw = strip_numeric(raw)   # Loại bỏ số
    return raw


def detectSpam(file_path):
    with open(file_path, 'r') as f:
        document = raw_text_preprocess(f.read())
        vector = np.zeros(len(dictionary))
        doc = document.split(" ")
        for word in doc:
            if word in dictionary:
                vector[dictionary.index("{}".format(word))] += 1
    vector = np.reshape(vector, (1, -1))
    result = model.predict(vector)
    return result[0]


