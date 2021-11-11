import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer


import tkinter as tk  
from tkinter import ttk  
from tkinter import Menu  
from tkinter import messagebox as mbox  
app = tk.Tk()



data = pd.read_csv("data.csv")

print(data.head())

print(data.info())

print(data.isnull().sum())


print(data.head())

print(data['Label'].value_counts())

data['Body'].fillna('ok', inplace=True)

data['Label'].fillna(0, inplace=True)


print(data)

X_train, X_test, y_train, y_test = train_test_split(data['Body'], data['Label'],test_size = 0.2, random_state=0, shuffle=True, stratify=data['Label'])

print(y_test.shape)

print(X_train)

print(y_train)

print("Training Started")

from sklearn.ensemble import VotingClassifier
# Fitting RF and LM to the Training set
estimators = []
model1 = Pipeline([('tfidf', TfidfVectorizer()), ('clf', RandomForestClassifier(n_estimators=1000, n_jobs=-1))])
estimators.append(('rt1', model1))
model2 =Pipeline([('tfidf', TfidfVectorizer()), ('clf', RandomForestClassifier(n_estimators=1000, n_jobs=-1))])
estimators.append(('rt2', model2))
model3 =Pipeline([('tfidf', TfidfVectorizer()), ('clf', RandomForestClassifier(n_estimators=1000, n_jobs=-1))])
estimators.append(('rt3', model3))
model4 =Pipeline([('tfidf', TfidfVectorizer()), ('clf', RandomForestClassifier(n_estimators=1000, n_jobs=-1))])
estimators.append(('rt4', model4))
model5 =Pipeline([('tfidf', TfidfVectorizer()), ('clf', RandomForestClassifier(n_estimators=1000, n_jobs=-1))])
estimators.append(('rt5', model5))

#hybrid ensembling
ensemble = VotingClassifier(estimators)
ensemble.fit(X_train, y_train)
y_pred = ensemble.predict(X_test)


print("Training Ended")
print(y_pred)

#confussion Matrix
cm_HybridEnsembler = confusion_matrix(y_test, y_pred)
print("Confussion Matrix :")
print(cm_HybridEnsembler)
print(" ")
