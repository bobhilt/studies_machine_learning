#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 01:14:15 2018

@author: bobhilt
adapted from https://www.kdnuggets.com/2017/12/managing-machine-learning-workflows-scikit-learn-pipelines-part-1.html

"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import tree

iris_data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris_data)

# Create Pipelines
pipe_lr = Pipeline([('scl', StandardScaler()),
                    ('pca', PCA(n_components=2)),
                    ('clf', LogisticRegression(random_state=0))])

pipe_svm = Pipeline([('scl', StandardScaler()),
                     ('pca', PCA(n_components=2)),
                     ('clf', svm.SVC(random_state=0))])

pipe_dt = Pipeline([('scl', StandardScaler()),
                    ('pca', PCA(n_components=2)),
                    ('clf', tree.DecisionTreeClassifier(random_state=0))])

# List to help iterate
pipelines = [pipe_lr, pipe_svm, pipe_dt]

# Dictionary to help reference
pipe_dict = {0: 'Logistic Regression', 
             1: 'Support Vector Machine',
             2: 'Decision Tree'}

# Fit the pipelines
for pipe in pipelines:
    pipe.fit(X_train, y_train)
    
# Compare accuracy
best_acc = 0.0
best_clf = 0
best_pipe = ''

for idx, val in enumerate(pipelines):
    pipe_score = val.score(X_test, y_test)
    print('%s pipeline test accuracy: %.3f' % (pipe_dict[idx], pipe_score))

    if pipe_score > best_acc:
        best_acc = pipe_score
        best_pipe = val
        best_clf = idx

print('Classifier with best accuracy: %s' % pipe_dict[best_clf])

# Save pipeline to file
joblib.dump(best_pipe, 'best_pipeline.pkl', compress=1)
print('Saved %s pipeline to file' % pipe_dict[best_clf])        
        
    