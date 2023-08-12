import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from keras.models import Sequential
from keras.layers import Dense
import keras.activations,keras.losses,keras.metrics

data=pd.read_csv('health care diabetes.csv')
print(data.columns)
print(data.info())
print(data.isna().sum())
print(data.describe())


'''for i in data.columns.values:
    sn.boxplot(data[i])
    plt.show()'''
#blood_pressure
#insulin
#BMI
#diaPedigreefunciton
#AGE
print(len(data))
threshold=2
for i in data.columns.values:
    upper=data[i].mean() + threshold*data[i].std()
    lower=data[i].mean() - threshold*data[i].std()
    data=data[(data[i]>lower)&(data[i]<upper)]

print(len(data))

'''for i in data.columns.values:
    sn.boxplot(data[i])
    plt.show()'''

for i in data.columns.values:
        print(data[i].value_counts())
        print(data[i].value_counts().index)

'''for i in  data.columns.values:
    if len(data[i].value_counts )<=5:
        sn.countplot(data[i])
        plt.show()

plt.figure(figsize=(17, 6))
corr = data.corr(method='spearman')
my_m = np.triu(corr)
sn.heatmap(corr, mask=my_m, annot=True, cmap="Set2")
plt.show()

for i in data.select_dtypes(include='number').columns.values:
    for j in data.select_dtypes(include='number').columns.values:
        sn.distplot(data[i], label=f"{i}", color='red')
        sn.distplot(data[j], label=f"{j}", color="blue")
        plt.title(f"ITS {i} vs {j}")
        plt.legend()
        plt.show()

for i in data.select_dtypes(include='number').columns.values:
    for j in data.select_dtypes(include='number').columns.values:
        sn.histplot(data[i], label=f"{i}", color='red')
        sn.histplot(data[j], label=f"{j}", color="blue")
        plt.title(f"ITS {i} vs {j}")
        plt.legend()
        plt.show()'''

x=data[['Pregnancies','Glucose','BloodPressure','BMI','DiabetesPedigreeFunction','Age']]
y=data['Outcome']
x_train,x_test,y_train,y_test=train_test_split(x,y)
lr = LogisticRegression(max_iter=200)
lr.fit(x_train, y_train)
print('The logistic regression: ', lr.score(x_test, y_test))

xgb = XGBClassifier()
xgb.fit(x_train, y_train)
print("the Xgb : ", xgb.score(x_test, y_test))

lgb = LGBMClassifier()
lgb.fit(x_train, y_train)
print('The LGB', lgb.score(x_test, y_test))

tree = DecisionTreeClassifier(criterion='gini', max_depth=1)
tree.fit(x_train, y_train)
print('Dtree ', tree.score(x_test,y_test))

rforest = RandomForestClassifier(criterion='gini')
rforest.fit(x_train, y_train)
print('The random forest: ', rforest.score(x_test, y_test))

adb = AdaBoostClassifier()
adb.fit(x_train, y_train)
print('the adb ', adb.score(x_test, y_test))

grb = GradientBoostingClassifier()
grb.fit(x_train, y_train)
print('Gradient boosting ', grb.score(x_test, y_test))

bag = BaggingClassifier()
bag.fit(x_train, y_train)
print('Bagging', bag.score(x_test, y_test))

Y=pd.get_dummies(data['Outcome'])
x_trin,x_tst,y_trin,y_tst=train_test_split(x,Y)

models=Sequential()
models.add(Dense(units=x.shape[1],input_dim=x.shape[1],activation=keras.activations.sigmoid))
models.add(Dense(units=x.shape[1],activation=keras.activations.sigmoid))
models.add(Dense(units=x.shape[1],activation=keras.activations.sigmoid))
models.add(Dense(units=x.shape[1],activation=keras.activations.sigmoid))
models.add(Dense(units=x.shape[1],activation=keras.activations.sigmoid))
models.add(Dense(units=Y.shape[1],activation=keras.activations.sigmoid))
models.compile(optimizer='adam',loss=keras.losses.binary_crossentropy,metrics='accuracy')
hist=models.fit(x_trin,y_trin,batch_size=20,epochs=350)

plt.plot(hist.history['accuracy'], label='training accuracy', marker='o', color='red')
plt.plot(hist.history['loss'], label='loss', marker='o', color='darkblue')
plt.title('Training Vs  Validation accuracy with adam optimizer')
plt.legend()
plt.show()


models1=Sequential()
models1.add(Dense(units=x.shape[1],input_dim=x.shape[1],activation=keras.activations.sigmoid))
models1.add(Dense(units=x.shape[1],activation=keras.activations.sigmoid))
models1.add(Dense(units=x.shape[1],activation=keras.activations.sigmoid))
models1.add(Dense(units=x.shape[1],activation=keras.activations.sigmoid))
models1.add(Dense(units=x.shape[1],activation=keras.activations.sigmoid))
models1.add(Dense(units=Y.shape[1],activation=keras.activations.sigmoid))
models1.compile(optimizer='rmsprop',loss=keras.losses.binary_crossentropy,metrics='accuracy')
histo=models1.fit(x_trin,y_trin,batch_size=20,epochs=350)

plt.plot(histo.history['accuracy'], label='training accuracy', marker='o', color='red')
plt.plot(histo.history['loss'], label='loss', marker='o', color='darkblue')
plt.title('Training Vs  Validation accuracy with adam rmsprop')
plt.legend()
plt.show()