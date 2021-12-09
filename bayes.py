import pandas as pd
import numpy as np
from gensim.parsing.preprocessing import strip_non_alphanum, strip_multiple_whitespaces,preprocess_string, split_alphanum, strip_short, strip_numeric
import re
import csv


def raw_text_preprocess(raw):
    raw = re.sub(r"http\S+", "", raw)  #bỏ đường dẫn
    raw = strip_non_alphanum(raw).lower().strip()   #loại bỏ kí tự lạ, chuẩn hóa
    raw = split_alphanum(raw)   #tách các từ vô nghĩa
    raw = strip_short(raw, minsize=2)   # loại bỏ kí tự đứng 1 mình
    raw = strip_numeric(raw)   # Loại bỏ số 

    return raw

xls_file = pd.read_csv('emails.csv')
#dfs = xls_file.parse('Sheet1')

document = []
label = []

for d in xls_file.Document:
    document.append(d)
for l in xls_file.Label:
    label.append(l)

document = [raw_text_preprocess(d) for d in document]


#xay dung bag of words

print("xaydung bagwwords")
set_words = []

with open('emailss.csv') as f:
    # nCols = len(f.readline().split(','))
    reader = csv.reader(f)
    row1 = next(reader)
# print(row1)
r = row1[1:3001]
set_words = r
#print (set_words)

#for doc in document:
#    words = doc.split(' ')
#    set_words += words
#    set(set_words)

#print("Kich thuoc BAG WORDS:" , len(set_words))



#Chuyển mail sang vector
vectors = []

for doc in document:
    vector = np.zeros(len(set_words))
    for i, word in enumerate(set_words):
        if word in doc:
            vector[i] += 1
    vectors.append(vector)

print("So vector, so chieu: ",np.shape(vectors))

def smoothing(a, b):
    return float((a+1)/(b+1))

# P(spam), P(nonspam)
spam = 0
non_spam = 0
for l in label:
    if l == 1:
        spam += 1
    else:
        non_spam += 1
print("So thu Spam, Nonspam: ",spam, non_spam)

spam_coef = smoothing(spam, (spam+non_spam))
non_spam_coef = smoothing(non_spam, (spam+non_spam))

bayes_matrix = np.zeros((len(set_words), 4)) #app/spam, app/nonspam, nonapp/spam, nonapp/nonspam

for i, word in enumerate(set_words):
    app_spam = 0
    app_nonspam = 0
    nonapp_spam = 0
    nonapp_nonspam = 0
    for k, v in enumerate(vectors):
        if v[i] == 0:
            if label[k] == 1:
                nonapp_spam += 1
            else:
                nonapp_nonspam += 1
            
        else:
            if label[k] == 1:
                app_spam += 1
            else:
                app_nonspam += 1
            
                
    bayes_matrix[i][0] = smoothing(app_spam, spam)
    bayes_matrix[i][1] = smoothing(app_nonspam, non_spam)
    bayes_matrix[i][2] = smoothing(nonapp_spam, spam)
    bayes_matrix[i][3] = smoothing(nonapp_nonspam, non_spam)

def compare(predict_spam, predict_non_spam, log):
    while (log[0] > log[1]):
        predict_spam /= 10
        log[0] -=1
        if predict_spam > predict_non_spam:
            return True

    while(log[1] > log[0]):
        predict_non_spam /= 10
        log[1] -= 1
        if predict_non_spam > predict_spam:
            return False
        
    if predict_spam > predict_non_spam:
        return True
    return False

def predict(mail):
    mail = raw_text_preprocess(mail)
    
    vector = np.zeros(len(set_words))
    for i, word in enumerate(set_words):
        if word in mail:
            vector[i] = 1
    log = np.zeros(2)

    predict_spam = spam_coef
    predict_non_spam = non_spam_coef

    for i, v in enumerate(vector):
        if v == 0:
            predict_spam *= bayes_matrix[i][2]
            predict_non_spam *= bayes_matrix[i][3]
        else:
            predict_spam *= bayes_matrix[i][0]
            predict_non_spam *= bayes_matrix[i][1]

        if predict_spam < 1e-10:
            predict_spam *= 1000
            log[0] += 1

        if predict_non_spam < 1e-10:
            predict_non_spam *= 1000
            log[1] +=1
            
    if compare(predict_spam, predict_non_spam, log):
        return 1
    return 0


print("TESTTTTT")
test = pd.ExcelFile('./Book2.xlsx')
df_test = test.parse('Sheet1')
document_test = []
label_test = []


for d in df_test.Document:
    document_test.append(d)
    #print(predict(d))
for l in df_test.Label:
    label_test.append(l)    
#######################################




pred = [predict(d) for d in document_test]
print(accuracy_score(pred, label_test))

from sklearn.model_selection import train_test_split
X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)