import re
import csv
import numpy as np
import pandas as pd
from gensim.parsing.preprocessing import strip_non_alphanum, split_alphanum, strip_short, strip_numeric
from sklearn.naive_bayes import MultinomialNB

with open('emailstrain.csv') as f:
    reader = csv.reader(f)
    row1 = next(reader)
    r = row1[1:3001]

dictionary = r

# Tap du lieu
train_file = pd.read_csv("emailstrain.csv")
train_matrix = train_file.iloc[:, 1:-1].values
train_label = train_file.iloc[:, -1].values

# Train
model = MultinomialNB()
model.fit(train_matrix, train_label)
