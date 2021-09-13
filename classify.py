
import pandas as pd 
import numpy as np
from sklearn.feature_extraction.text import stop_words
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle

class classify_news():
	def __init__(self, text):
		self.text = text
	def classify(self):
		classifyer_name = 'pac.pkl'
		pac = pickle.load(open(classifyer_name, 'rb'))
		vectorizer_name = 'tfidf_vectorizer.pkl'
		vectorizer = pickle.load(open(vectorizer_name, 'rb'))
		pred = pac.predict(vectorizer.transform([self.text]))
		print(pred[0])
