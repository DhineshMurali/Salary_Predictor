import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

data  = pd.read_csv('Engineering_graduate_salary.csv')
#print(data.head(10))

print(data['Degree'].unique())

#Degree
data['Degree'] = data['Degree'].map({'B.Tech/B.E.': 1, 'M.Tech./M.E.': 2, 'MCA': 3, 'M.Sc. (Tech.)': 4}).astype(int)
#Gender
data['Gender'] = data['Gender'].map({'m': 0, 'f': 1}).astype(int)
#Specialization
#vectoriser = CountVectorizer()
#data['Specialization'] = vectoriser.fit_transform(data['Specialization'])
#data['Specialization'] = data['Specialization'].astype(float)

#Checking for missing data
sns.heatmap(data.isnull())
plt.show()

#Train Test Split

X = pd.DataFrame(np.c_[data['Gender'],data['10percentage'],data['12percentage'],data['CollegeTier'],data['Degree'],data['collegeGPA'],data['GraduationYear'],data['Domain'],data['agreeableness'],data['openess_to_experience']],columns=['Gender','10percentage','12percentage','CollegeTier','Degree','collegeGPA','GraduationYear','Domain','agreeableness','openess_to_experience'])
Y = pd.DataFrame(data['Salary'])

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=30)

model = LinearRegression()
model.fit(X_train,Y_train)

regressor = DecisionTreeClassifier(random_state=0)

regressor.fit(X_train, Y_train)

prediction = regressor.predict(X_test)
#prediction = model.predict(X_test)
#print(prediction)

error = np.sqrt(metrics.mean_absolute_error(Y_test,prediction))
print(error)
