import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

a=pd.read_csv(r"D:\DATA SCIENCE\data science\2. Logistic-Regression\titanic_train.csv")
print(a.info())
print(a.head())
print(a['Cabin'].isnull())

sns.heatmap(a.isnull())
plt.title('Null values in dataset')
plt.show()

print(a.groupby('Sex')['Age'].describe())

print(a.groupby('Pclass')['Age'].describe())


a.drop(['PassengerId','Name','Embarked','Cabin','Ticket'],axis=1,inplace=True)

print(a)

sns.countplot(x='Survived', data=a)
plt.title('Survival Distribution')
plt.show()

sns.countplot(x='Survived', hue='Sex', data=a)
plt.title('Survival Count by Gender')
plt.show()

sns.barplot(x='Pclass', y='Survived', hue='Sex', data=a)
plt.title('Survival Rate by Passenger Class')
plt.show()

sns.boxplot(x='Pclass', y='Age', data=a)
plt.title('Age Distribution by Passenger Class')
plt.show()

sns.violinplot(x='Survived', y='Fare', data=a)
plt.title('Distribution of Fare for Survived and Not Survived')
plt.show()

sns.violinplot(x='Survived', y='Age', data=a)
plt.title('Age Distribution by Survival')
plt.show()

sns.distplot(a['Age'].dropna(),kde=False,color='darkred',bins=5)
plt.title('Count of Age')
plt.show()

sns.pairplot(a[['Age', 'SibSp', 'Parch', 'Fare']])
plt.suptitle('Pairplot of Numerical Features', y=1.02)
plt.show()

def sag(name):
    Age=name[0]
    Pclass=name[1]
    if pd.isnull(Age):
        if Pclass==1:
            return 38
        elif Pclass==2:
            return 29
        else:
            return 24
    else:
        return Age
    
    
a['Age']=a[['Age','Pclass']].apply(sag,axis=1)
a['Fare']=a['Fare'].round()

gender=pd.get_dummies(a['Sex'],drop_first=True)

gender=gender.astype(int)


print(gender)

a.drop('Sex',axis=1,inplace=True)


a=pd.concat([a,gender],axis=1)

print(a.corr())

print(a.info())
print(a.head())


x=a[['Pclass','Age','SibSp','Parch','Fare','male']]
y=a['Survived']


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=35)


from sklearn.linear_model import LogisticRegression

logmodel=LogisticRegression()
logmodel.fit(x_train,y_train)

print(logmodel.coef_)


y_pred=logmodel.predict(x_test)
print(y_pred)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_pred))


(142+82)/(142+82+30+14)

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


from sklearn.metrics import RocCurveDisplay
RocCurveDisplay.from_estimator(logmodel,x_test,y_test)
plt.show()



from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
z=scaler.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(z,y,test_size=0.30,random_state=35)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
print('decision tree accuracy is: ',cross_val_score(dt,z,y,cv=10,scoring='accuracy').mean())

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
print('random forest accuracy is: ',cross_val_score(rfc,z,y,cv=10,scoring='accuracy').mean())

from sklearn.naive_bayes import GaussianNB
nb= GaussianNB()
print('naive bayes accuracy is: ',cross_val_score(nb,z,y,cv=10,scoring='accuracy').mean())

from sklearn.svm import SVC
svm = SVC()
print('svm accuracy is: ',cross_val_score(svm,z,y,cv=10,scoring='accuracy').mean())


from sklearn.neighbors import KNeighborsClassifier
error_rate = []
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train, y_train)
    pred_i = knn.predict(x_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()


knn = KNeighborsClassifier(n_neighbors=15)
print('knn accuracy is: ',cross_val_score(knn,z,y,cv=10,scoring='accuracy').mean())



