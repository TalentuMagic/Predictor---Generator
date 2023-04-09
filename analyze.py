import warnings
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import tensorflow as tf
dataset = pd.read_csv('dataSet.csv', delimiter=',')
# Step 2: Separate features and target
X = dataset.iloc[:, 17:].values
# y contains only the last column of the dataSet, resulting only the target values
y = dataset.iloc[:, -9].values
# Split the dataset into training and testing sets - 25% test, 75% train
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

# Standardize the features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Define the models to train on the dataSet
perceptron = Perceptron()
naive_bayes = GaussianNB()
decision_tree = DecisionTreeClassifier()
mlp = MLPClassifier()

kerasModel = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(8,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='softmax')
])
kerasModel.compile(loss='categorical_crossentropy',
                   optimizer='adam', metrics=['accuracy'])
# Train the models
perceptron.fit(X_train, y_train)
naive_bayes.fit(X_train, y_train)
decision_tree.fit(X_train, y_train)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    mlp.fit(X_train, y_train)

kerasModel.fit(X_train, y_train, epochs=50, batch_size=32)
# Make predictions
perceptron_pred = perceptron.predict(X_test)
naive_bayes_pred = naive_bayes.predict(X_test)
decision_tree_pred = decision_tree.predict(X_test)
mlp_pred = mlp.predict(X_test)
# Evaluate the Keras Neural-Network model
test_loss, test_acc = kerasModel.evaluate(X_test, y_test)
print('Keras Neural-Network Test accuracy:', round(test_acc, 7), '\n')

# Evaluate the models
print('Perceptron:')
print(confusion_matrix(y_test, perceptron_pred))
print(classification_report(y_test, perceptron_pred))

print('Naive Bayes:')
print(confusion_matrix(y_test, naive_bayes_pred))
print(classification_report(y_test, naive_bayes_pred))

print('Decision Tree:')
print(confusion_matrix(y_test, decision_tree_pred))
print(classification_report(y_test, decision_tree_pred))

print('MLP:')
print(confusion_matrix(y_test, mlp_pred))
print(classification_report(y_test, mlp_pred))
