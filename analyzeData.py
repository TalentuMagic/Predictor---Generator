import matplotlib.pyplot as mtp
import joblib
import pandas
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.linear_model import SGDClassifier, SGDRegressor, Ridge
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, r2_score
import seaborn

dataSet = pandas.read_csv("dataSet.csv")


def mFunc(X, y, model):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = model
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print('Prediction Score:', model.score(X_test, y_test))

    return X_train, X_test, y_train, y_test, model, predictions


def predictFunc():
    global dataSet

    # inputs
    X = dataSet[['Good_1', 'Good_2', 'Good_3', 'Good_4',
                 'Good_5', 'Good_6', 'Good_7', 'Good_8']]
    # target
    y = dataSet['Winning Chance']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(predictions)
    print(model.score(X_test, y_test))
    # mtp.figure(figsize=(8, 8))
    # seaborn.heatmap(predictions, annot=True)
    # mtp.xlabel('Predicted')
    # mtp.ylabel('Truth')
    # mtp.show()


def predictLine1():
    global dataSet

    X = dataSet[["Good_1", "Good_2", "Good_3", "Good_4", "Good_5", "Good_6", "Good_7", "Good_8",
                 "Winning Chance", "Line1As", "Line2As", "Line3As", "Line4As", "Line5As", "Line6As", "Line7As", "Line8As"]]
    y = dataSet[['Line1As', 'Good_1']]

    X_train, X_test, y_train, y_test, model, predictions = mFunc(
        X=X, y=y, model=DecisionTreeRegressor())

    print(model.predict(X_test[:1]))


predictLine1()
