# Importing Libraries
import streamlit as st
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

st.markdown('<style>h1{color: blue;}</style>', unsafe_allow_html=True)

st.title('Heart Disease Prediction App')
st.markdown('<style>h3{color: red;}</style>', unsafe_allow_html=True)
st.subheader(''' 
 **Let's try and see which one is the Best!**
---
''')
st.sidebar.header('''Heart Disease Dataset
---
''')
st.sidebar.subheader("Select Classifer for Prediction")
clf_name = st.sidebar.selectbox(
    " ",
    ('Logistic Regression','SVM','Naive Bayes',
    'Gradient Boosting','Random Forest','Decision Tree'))
# Data Pre-processing
df = pd.read_csv('heart.csv')

predictors = df.drop('target',1)
target = df['target']

st.write('Shape of Dataset:',df.shape)
st.write('Number of classes:', len(np.unique(target)))

def add_params(clf_name):
    params = dict()
    if clf_name == 'SVM':
        C = st.sidebar.slider('C',1.00,15.0,2.0)
        params['C']= C
        kernel = st.sidebar.selectbox("Kernel",('linear', 'poly', 'rbf','sigmoid'))
        params['kernel'] = kernel
    elif clf_name == 'Random Forest':
        md = st.sidebar.slider('max_depth',2,15,2)
        params['max_depth'] = md
        n_es = st.sidebar.slider('n_estimators',1,1000,100)
        params['n_estimators'] = n_es
    elif clf_name == 'Decision Tree':
        md = st.sidebar.slider('max_depth',2,15,2)
        params['max_depth'] = md
        sp = st.sidebar.selectbox("Splitter",('best','random'))
        params['splitter'] = sp
    elif clf_name == 'Logistic Regression':
        C = st.sidebar.slider('C',1.0,15.0,2.0)
        params['C']= C
    elif clf_name == 'Gradient Boosting':
        loss = st.sidebar.selectbox("loss",('deviance','exponential'))
        params['loss'] = loss
        n_es = st.sidebar.slider('n_estimators',1,1000,100)
        params['n_estimators'] = n_es
    
    return params

params = add_params(clf_name)

def get_clf(clf_name,params):
    clf = None
    if clf_name == 'SVM':
        clf = svm.SVC(C=params['C'],kernel=params['kernel'])
    elif clf_name == 'Random Forest':
        clf = RandomForestClassifier(n_estimators=params['n_estimators'], 
            max_depth=params['max_depth'],random_state=2000)
    elif clf_name == 'Decision Tree':
        clf = DecisionTreeClassifier(splitter=params['splitter'],
        max_depth=params['max_depth'],random_state=200)
    elif clf_name == 'Logistic Regression':
        clf = LogisticRegression(C=params['C'])
    elif clf_name == 'Gradient Boosting':
        clf = GradientBoostingClassifier(loss=params['loss'],
        n_estimators=params['n_estimators'],random_state=100)
    else:
        clf = GaussianNB()
    
    return clf

clf = get_clf(clf_name,params)

# Classification
x_train,x_test,y_train,y_test = train_test_split(predictors,target,
                                test_size = 0.2)

clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)

accuracy = round(accuracy_score(y_test,y_pred),2)*100

st.write(f'CLassifier = {clf_name}')
st.write(f'Accuracy = {accuracy}')

if st.sidebar.button("Know About Developer"):
    st.sidebar.header(''' Kuldeep Sharma aka [SoleCodr](https://github.com/SoleCodr) ''')
    st.sidebar.subheader('''GitHub Repository of this [App](https://github.com/SoleCodr/Heart-disease-prediction-app) ''')
    st.balloons()

