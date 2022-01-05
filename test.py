from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from main import *


# Tạo tập test
test_file = pd.read_csv('emailstest.csv')
test_document = []
test_label = []

for d in test_file.Document:
    test_document.append(d)
for l in test_file.Label:
    test_label.append(l)

document = [raw_text_preprocess(d) for d in test_document]

# Chuyển mail sang vector
test_matrix = []

for doc in document:
    words = doc.split(" ")
    vector = np.zeros(len(dictionary))
    for word in words:
        if word in dictionary:
            vector[dictionary.index("{}".format(word))] += 1
    test_matrix.append(vector)

# test
result = model.predict(test_matrix)
print(confusion_matrix(test_label, result))
print(accuracy_score(test_label, result))
