from django.shortcuts import render
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


def home(request):
    return render(request,"home.html")

def predict(request):
    return render(request,"predict.html")

def result(request):
    data = pd.read_csv('D:/ML Projects/Django + ML prediction/Engineering_graduate_salary.csv')
    # Degree
    data['Degree'] = data['Degree'].map({'B.Tech/B.E.': 1, 'M.Tech./M.E.': 2, 'MCA': 3, 'M.Sc. (Tech.)': 4}).astype(int)
    # Gender
    data['Gender'] = data['Gender'].map({'m': 0, 'f': 1}).astype(int)
    # Train Test Split

    X = pd.DataFrame(np.c_[data['Gender'], data['10percentage'], data['12percentage'], data['CollegeTier'], data[
        'Degree'], data['collegeGPA'], data['GraduationYear'], data['Domain'], data['agreeableness'], data[
                               'openess_to_experience']],
                     columns=['Gender', '10percentage', '12percentage', 'CollegeTier', 'Degree', 'collegeGPA',
                              'GraduationYear', 'Domain', 'agreeableness', 'openess_to_experience'])
    Y = pd.DataFrame(data['Salary'])

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=30)

    model = DecisionTreeClassifier()
    model.fit(X_train, Y_train)

    var1 = float(request.GET['n1'])
    var2 = float(request.GET['n2'])
    var3 = float(request.GET['n3'])
    var4 = float(request.GET['n4'])
    var5 = float(request.GET['n5'])
    var6 = float(request.GET['n6'])
    var7 = float(request.GET['n7'])
    var8 = float(request.GET['n8'])
    var9 = float(request.GET['n9'])
    var10 = float(request.GET['n10'])

    pred = model.predict(np.array([var1,var2,var3,var4,var5,var6,var7,var8,var9,var10]).reshape(1,-1))
    #pred = round(pred[0])

    salary = "The predicted salary is $"+str(pred)
    return render(request,"predict.html",{"result2":salary})