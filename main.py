import streamlit as st
from sklearn import datasets

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

import numpy as np

st.title("Streamlit example")

st.write("""
# Explore different Classifier
""")


dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine Dataset"))

classifier_name = st.sidebar.selectbox("Select Classifier", ("KNN", "SVM", "Random Forest"))

@st.cache(suppress_st_warning=True) 
def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    x = data.data
    y = data.target
    return x,y

X,y = get_dataset(dataset_name)

st.write("The shape of the dataset", X.shape)
st.write("Number of classes", len(np.unique(y)))


def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "KNN":
        K = st.sidebar.slider("K",1,15)
        params["K"] = K
    elif clf_name == "SVM":
        C = st.sidebar.slider("C",.01,10.0)
        params["C"] = C
    else:
        R = st.sidebar.slider("max_depth",2,15)
        n_estimators = st.sidebar.slider("No. of Estimators", 1, 100)
        params["R"] = R
        params["N"] = n_estimators
    return params

params = add_parameter_ui(classifier_name)


def get_classifier(clf_name, params):
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif clf_name == "SVM":
        clf = SVC(C=params["C"])
    else:
        clf = RandomForestClassifier(n_estimators=params["N"], max_depth=params["R"], random_state=1234)
    return clf

clf = get_classifier(classifier_name, params)

xtrain, xtest, ytrain, ytest = train_test_split(X,y, test_size=0.2, random_state=1234)

clf.fit(xtrain, ytrain)
pred = clf.predict(xtest)

acc = accuracy_score(ytest, pred)

st.write(f"Classifier = {classifier_name}")
st.write(f"Accuracy = {acc*100}%")


pca = PCA(2)

xpro = pca.fit_transform(X)

x1 = xpro[:,0]
x2 = xpro[:,1]

fig = plt.figure()
plt.scatter(x1,x2,c=y,alpha=.8,cmap="viridis")
plt.xlabel("Principle Component 1")
plt.ylabel("Principle Component 2")

plt.colorbar()

st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()