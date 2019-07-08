import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score,classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

#reading data
df=pd.read_csv("/..../enter your path of dataset/...csv")

#making ham=0,spam=1 used as target matrix in label_num column
df['label_num'] = df.label.map({'ham':0,'spam':1})
features = df["text"]
labels=df["label_num"]

#plotting data
sn.countplot(df["label"])
print(plt.show())

#spliting training and testing data
f_train,f_test,l_train,l_test= train_test_split(features,labels,test_size=.1)
f_train=np.array(f_train)
f_test=np.array(f_test)
l_train=np.array(l_train)
l_test=np.array(l_test)

#inorder to train model we need to convert text to appropriate numerical values using vetorization

vect = CountVectorizer()
f_train_count = vect.fit_transform(f_train)
f_test_count= vect.transform(f_test)
#trans = TfidfTransformer()
#f_train_trans = trans.fit_transform(f_train_count)
#f_test_trans = trans.transform(f_test_count)


#model training and prediction
sd = MultinomialNB()
sd.fit(f_train_count,l_train)
print("\n\n\033[1;32;40m [+]train score: ",sd.score(f_train_count,l_train))
print("\n\n[+]test score:  ",sd.score(f_test_count,l_test))
y_class = sd.predict(f_test_count)
a= accuracy_score(l_test,y_class)
d= precision_score(l_test,y_class)
e= recall_score(l_test,y_class)
g= f1_score(l_test,y_class)
print("\n\n[+]accuracy score = " ,a)
print("\n\n[+]precision score = ",d)
print("\n\n[+]recall score = ", e)
print("\n\n[+]f1 score = ", g)
report = classification_report(y_class,l_test)
print("\n",report)


#testing with data other than in dataset
print("\n\n[+] lets test with other unique msgs other than the datasets used: ") 
t=[input("[+]enter a text msg to test : ")]
t=np.array(t)
t=vect.transform(t)
prediction = sd.predict(t)
if prediction == 0:
    print("HAAM!")
else:
    print("\033[1;31;40m [+]SPAAAM!")
