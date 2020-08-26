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
st.markdown('<style>h3{color: red;}</style>', unsafe_allow_html=True)

st.title('Heart Disease Prediction App')
st.subheader(''' 
 **Let's try and see which one is the Best!**
---
''')
st.markdown(''' 
[Data-Set can be download from here.](https://raw.githubusercontent.com/SoleCodr/Heart-disease-prediction-app/master/heart.csv)
''')
st.sidebar.header('''Heart Disease Dataset
---
''')
st.sidebar.subheader("Select Classifer for Prediction")
clf_name = st.sidebar.selectbox(
    " ",
    ('Logistic Regression','SVM','Naive Bayes',
    'Gradient Boosting','Random Forest','Decision Tree'))

opt = st.sidebar.selectbox(" Do you want to have custom parameters?",('Yes','No'))

if opt == 'Yes':
    
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
else:
    def get_clf(clf_name):
        clf = None
        if clf_name == 'SVM':
            clf = svm.SVC()
        elif clf_name == 'Random Forest':
            clf = RandomForestClassifier()
        elif clf_name == 'Decision Tree':
            clf = DecisionTreeClassifier()
        elif clf_name == 'Logistic Regression':
            clf = LogisticRegression()
        elif clf_name == 'Gradient Boosting':
            clf = GradientBoostingClassifier()
        else:
            clf = GaussianNB()
        
        return clf
    clf = get_clf(clf_name)

# Data Pre-processing
df = pd.read_csv('heart.csv')
df = df.drop_duplicates()

predictors = df.drop('target',1)
target = df['target']

st.write('Shape of Dataset:',df.shape)
st.write('Number of classes:', len(np.unique(target)))

st.set_option('deprecation.showfileUploaderEncoding', False)
st.sidebar.markdown('### User Input:')

st.sidebar.markdown('''
[Example CSV input file](https://raw.githubusercontent.com/SoleCodr/Heart_Disease_Prediction/master/heart_example.csv)
''')
upld_file = st.sidebar.file_uploader("upload your input csv file", type=["csv"])
if upld_file is not None:
    df1 = pd.read_csv(upld_file)
else:
    def user_input():
        age = st.sidebar.slider("Age :",0,100,55)
        sex = st.sidebar.slider("Sex :",0,1,1)
        cp = st.sidebar.slider('cp :',0,3,2)
        testbps = st.sidebar.slider('testbps :',94,200,150)
        chol = st.sidebar.slider('chol :',126,564,250)
        fbs = st.sidebar.slider('fbs :',0,1,1)
        restecg = st.sidebar.slider('restecg :',0,2,1)
        thalach = st.sidebar.slider('thalach :',71,202,150)
        exang = st.sidebar.slider('exang :',0,1,1)
        oldpeak = st.sidebar.slider('oldpeak :',0.0,6.2,2.1)
        slope = st.sidebar.slider('slope :',0,2,1)
        ca = st.sidebar.slider('ca :',0,4,1)
        thal = st.sidebar.slider('thal :',1,3,2)
        
        inputs = { 'Age' : age,
                'sex' : sex,
                'cp' : cp,
                'testbps' : testbps,
                'chol' : chol,
                'testbps' : testbps,
                'chol' : chol,
                'fbs' : fbs,
                'restecg' : restecg,
                'thalach' : thalach,
                'exang'  : exang,
                'oldpeak' : oldpeak,
                'slope' : slope,
                'ca' : ca,
                'thal' : thal }
        features = pd.DataFrame(inputs, index = [0])
        return features

    df1 = user_input()

st.markdown("---")
st.write("Current Inputs are :",df1)
st.markdown("---")

# Classification
clf.fit(predictors,target)
y_pred = clf.predict(df1)
st.write(f'Classifier Name = {clf_name}')
st.subheader('Prediction')
predict = np.array(['0','1'])
pred = predict[y_pred]
if pred == '0':
    st.write('You are healthy')
else:
    st.write('You should consult a doctor')

st.sidebar.markdown('---')
if st.sidebar.button("Know About Developer"):
    st.sidebar.image('./img/self.jpg',use_column_width=True)
    st.sidebar.header(''' Kuldeep Sharma aka [SoleCodr](https://github.com/SoleCodr) ''')
    st.sidebar.subheader('''GitHub Repository of this [App](https://github.com/SoleCodr/Heart-disease-prediction-app) ''')
    st.balloons()

