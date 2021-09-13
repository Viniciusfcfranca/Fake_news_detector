import pandas as pd 
import numpy as np
from sklearn.feature_extraction import stop_words
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle



# Importing DataSet
df = pd.read_csv("news.csv")
df['label']= df['label'].map({"REAL": 0, "FAKE": 1})

#Seting the stop_words 
custom = stop_words.ENGLISH_STOP_WORDS
custom = list(custom)
common_unigrams = ['clinton', 'new', 'people', 'said', 'trump'] 
for i in common_unigrams:
    custom.append(i)

#spliting the dataset into train and test
X = df['text']
y= df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## Vectorizing
# The vectorizer was already chosen in the ipynb file called "Fake_news" recommended to check it for more details
vectorizer= TfidfVectorizer(stop_words=custom)
tftr = vectorizer.fit_transform(X_train) 
tft = vectorizer.transform(X_test)

## Classifying
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tftr ,y_train)

## Storing classifyer
classifyer_name = 'pac.pkl'
pickle.dump(pac, open(classifyer_name, 'wb'))

##Storing vectorizer
vectorizer_name = 'tfidf_vectorizer.pkl'
pickle.dump(vectorizer, open(vectorizer_name, 'wb'))

## Predicting and checking the accuracy
y_pred=pac.predict(tft)
score=accuracy_score(y_test,y_pred)

#Making confusion matrix
cnf_matriz = confusion_matrix(y_test,y_pred)

#Testing these models in a new dataset
ods = pd.read_csv('new_ds.csv')
n_t = ods['statement']
n_v = vectorizer.transform(n_t)
n_pred = pac.predict(n_v)

score=accuracy_score(ods['BinaryNumTarget'], n_pred)
print(score)
conf_mat = confusion_matrix(ods['BinaryNumTarget'], n_pred)
print(conf_mat) 


#Classifying new files
def classify(text):
	classifyer_name = 'pac.pkl'
	pac = pickle.load(open(classifyer_name, 'rb'))
	vectorizer_name = 'tfidf_vectorizer.pkl'
	vectorizer = pickle.load(open(vectorizer_name, 'rb'))
	pred = pac.predict(vectorizer.transform([text]))
	print(pred[0])


new_doc=["Photos show the U.S. service members killed in a 2021 suicide bombing in Kabul.",
         'In Afghanistan, “over 100 billion dollars spent on military contracts.”', 
         'women are "disproportionately" impacted by votes of male deputades',
         'The Trump administration worked to free 5,000 Taliban prisoners.']


for doc in new_doc:
    classify(doc)


















