import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split


from sklearn import metrics

df = pd.read_csv('adult.csv')

#data.info()

df.rename(columns={'capital-gain': 'capital gain', 'capital-loss': 'capital loss', 'native-country': 'country','hours-per-week': 'hours per week','marital-status': 'marital'}, inplace=True)

df.isin(['?']).sum(axis=0)
df['country'] = df['country'].replace('?',np.nan)
df['workclass'] = df['workclass'].replace('?',np.nan)
df['occupation'] = df['occupation'].replace('?',np.nan)

df.dropna(how='any',inplace=True)



df.drop(['educational-num','age', 'hours per week', 'fnlwgt', 'capital gain','capital loss', 'country'], axis=1, inplace=True)
df['income'] = df['income'].map({'<=50K': 0, '>50K': 1}).astype(int)

df['gender'] = df['gender'].map({'Male': 0, 'Female': 1}).astype(int)
#marital
df['marital'] = df['marital'].map({'Married-spouse-absent': 0, 'Widowed': 1, 'Married-civ-spouse': 2, 'Separated': 3, 'Divorced': 4,'Never-married': 5, 'Married-AF-spouse': 6}).astype(int)
#workclass
df['workclass'] = df['workclass'].map({'Self-emp-inc': 0, 'State-gov': 1,'Federal-gov': 2, 'Without-pay': 3, 'Local-gov': 4,'Private': 5, 'Self-emp-not-inc': 6}).astype(int)
#education
df['education'] = df['education'].map({'Some-college': 0, 'Preschool': 1, '5th-6th': 2, 'HS-grad': 3, 'Masters': 4, '12th': 5, '7th-8th': 6, 'Prof-school': 7,'1st-4th': 8, 'Assoc-acdm': 9, 'Doctorate': 10, '11th': 11,'Bachelors': 12, '10th': 13,'Assoc-voc': 14,'9th': 15}).astype(int)
#occupation
df['occupation'] = df['occupation'].map({ 'Farming-fishing': 1, 'Tech-support': 2, 'Adm-clerical': 3, 'Handlers-cleaners': 4,'Prof-specialty': 5,'Machine-op-inspct': 6, 'Exec-managerial': 7,'Priv-house-serv': 8,'Craft-repair': 9,'Sales': 10, 'Transport-moving': 11, 'Armed-Forces': 12, 'Other-service': 13,'Protective-serv':14}).astype(int)
#relationship
df['relationship'] = df['relationship'].map({'Not-in-family': 0, 'Wife': 1, 'Other-relative': 2, 'Unmarried': 3,'Husband': 4,'Own-child': 5}).astype(int)

df_x = pd.DataFrame(np.c_[df['relationship'], df['education'],df['occupation'],df['gender'],df['marital'],df['workclass']], columns = ['relationship','education','occupation','gender','marital','workclass'])

df_y = pd.DataFrame(df['income'])

#MODEL CREATION

reg = LogisticRegression()

x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.30, random_state=42)
reg.fit(x_train, y_train)

y_pred = reg.predict(x_test)

print(reg.predict([[1,7,7,0,2,0]]))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))