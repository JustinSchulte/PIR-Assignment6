import seaborn as sns; sns.set()  # for plot styling
from sklearn import svm
from sklearn import metrics
import pandas as pd

#limit rows because of time...
limitTrainRows = 10000
limitTestRows = 10000

df_train = pd.read_table('data/Fold1/train.txt',header=None, sep='\s+', index_col=False, nrows=limitTrainRows)
print("train amount: " + str(len(df_train)))
X_train = [[] for x in range(len(df_train))]
Y_train = []
for i in range(0,len(df_train)):
    print(i)
    relevance = 0
    if(df_train[0][i] > 0) :
        relevance = 1
    Y_train.append(relevance)
    for j in range(2, 52):
        X_train[i].append(df_train[j][i].split(':')[1])
print("Finished reading train-data...")

df_test = pd.read_table('data/Fold1/test.txt',header=None, sep='\s+', index_col=False, nrows=limitTestRows)
print("test amount: " + str(len(df_test)))
X_test = [[] for x in range(len(df_test))]
Y_test = []
for i in range(0,len(df_test)):
    print(i)
    relevance = 0
    if(df_test[0][i] > 0) :
        relevance = 1
    Y_test.append(relevance)
    for j in range(2, 52):
        X_test[i].append(df_test[j][i].split(':')[1])
print("Finished reading test-data...")

#LINEAR
#train model
clf = svm.SVC(kernel='linear', max_iter=50)
clf.fit(X_train, Y_train)
#test model
Y_predict = clf.predict(X_test)
prec_score = metrics.average_precision_score(Y_test, Y_predict)
recall_score = metrics.recall_score(Y_test, Y_predict)
print("linear_pre_score: " + str(prec_score))
print("linear_rec_score: " + str(recall_score))

#RBF
#train model
clf = svm.SVC(kernel='rbf', max_iter=50)
clf.fit(X_train, Y_train)
#test model
Y_predict = clf.predict(X_test)
prec_score = metrics.average_precision_score(Y_test, Y_predict)
recall_score = metrics.recall_score(Y_test, Y_predict)
print("rbf_pre_score: " + str(prec_score))
print("rbf_rec_score: " + str(recall_score))
