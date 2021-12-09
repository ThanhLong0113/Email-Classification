import re
import numpy as np
import csv
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from gensim.parsing.preprocessing import strip_non_alphanum, split_alphanum, strip_short, strip_numeric

# Tạo từ điển
with open('emailstrain.csv') as f:
    reader = csv.reader(f)
    row1 = next(reader)
    r = row1[1:3001]

dictionary = r

#Tạo tập train
train_file = pd.read_csv('./emailstrain.csv')
train_matrix = train_file.iloc[:, 1:-1].values
train_label = train_file.iloc[:, -1].values

#Train
model = MultinomialNB()
model.fit(train_matrix, train_label)

#chuẩn hóa mail

def raw_text_preprocess(raw):
    raw = re.sub(r"http\S+", "", raw)  #bỏ đường dẫn
    raw = strip_non_alphanum(raw).lower().strip()   #loại bỏ kí tự lạ, chuẩn hóa
    raw = split_alphanum(raw)   #tách các từ vô nghĩa
    raw = strip_short(raw, minsize=2)   # loại bỏ kí tự đứng 1 mình
    raw = strip_numeric(raw)   # Loại bỏ số 

    return raw

#Tạo tập test
test_file = pd.read_csv('emailstest.csv')
test_document = []
test_label = []

for d in test_file.Document:
    test_document.append(d)
for l in test_file.Label:
    test_label.append(l)

document = [raw_text_preprocess(d) for d in test_document]

#Chuyển mail sang vector
test_matrix = []

for doc in document:
    vector = np.zeros(len(dictionary))
    for i, word in enumerate(dictionary):
        if word in doc:
            vector[i] += 1
    test_matrix.append(vector)

#test
result = model.predict(test_matrix)
print(result)
print(confusion_matrix(result, test_label))
print(accuracy_score(result, test_label))