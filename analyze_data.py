import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from tqdm import tqdm
from datetime import datetime
import asyncio
import numpy as np

while (True):
    line = int(
        input("On what line should we train the model?\nSelect Even lines from 2-8\n"))
    if line % 2 == 0 and line in range(2, 8, 2):
        break
    else:
        raise SystemExit("WRONG SELECTION! RUN THE PROGRAM AGAIN")
# load the dataset's metadata
dataset = pd.read_csv("dataSet.csv")

# defining the label & ordinal encoders to encode the inputs and outputs into usable data
labelEncoder = LabelEncoder()
ordinalEncoder = OrdinalEncoder()
# defining the input data and outputs
# taking the first 17 columns from the dataset and duplicates (if any)
X = np.array(dataset.iloc[:, :17].drop_duplicates())
# encoding the features for the inputs
X = ordinalEncoder.fit_transform(X)
# categorizing the inputs in a categorical matrix
X = tf.keras.utils.to_categorical(X)
# transposing the input matrix to have the same shape as the output
# print(X.shape)
X = X.reshape(48600, -1)

# taking the even lines 2-8 from the dataset as outputs
y = np.array(dataset.iloc[:, line])

# encoding the outputs
y = labelEncoder.fit_transform(y)
# categorizing the outputs in a categorical matrix
y = tf.keras.utils.to_categorical(y)
# print(y.shape)

# taking the number of labels available for both input and output data
num_labels_X = X.shape[1]
num_labels_y2 = y.shape[1]

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# define the Neural Network model
model = tf.keras.Sequential()
# start with num_labels_X neurons (in this case 2142), with ReLU activation and a L2 regularizer to prevent overfitting
model.add(tf.keras.layers.Dense(num_labels_X, activation='relu', input_shape=(
    num_labels_X,), kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(tf.keras.layers.Dropout(0.2))
# the next two layers consists of a divisor of the num_labels_X
model.add(tf.keras.layers.Dense(156, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
# the output layer has the num_labels_y2 (in this case 56)
model.add(tf.keras.layers.Dense(num_labels_y2, activation='softmax'))

# compiling the model
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

num_epochs = 5  # Start with a moderate number of epochs and adjust if needed
batch_size = 128  # Adjust based on available memory and model complexity

# each time the model gets updated, the checkpointer updated the .hdf5 file (Hierarchical Data Format)
checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=f'./analysed_Data_line{line}.hdf5',
                                                  verbose=1, save_best_only=True)
# fit the model according to the neural network and the configurations done above
model.fit(X2_train, y2_train, batch_size=batch_size, epochs=num_epochs,
          validation_data=(X2_test, y2_test), callbacks=[checkpointer], verbose='1')

predict = model.predict(X2_test)
classes_X = np.argmax(predict, axis=1)
class_X = np.argmax(classes_X)
label_y = labelEncoder.inverse_transform([class_X])
print(label_y)


def predictLine(inputLine: str):
    X_predict = np.array([[inputLine]])
    print(X_predict, X_predict.shape)
    X_predict = LabelEncoder().fit_transform(X_predict)
    X_predict = tf.keras.utils.to_categorical(X_predict)

    y_pred = model.predict(X_predict)
    predicted_classes = np.argmax(y_pred, axis=1)
    predicted_class = np.argmax([predicted_classes])
    print(predicted_class)
    predicted_label = labelEncoder.inverse_transform(predicted_class)
    print(predicted_label)


predictLine("**$$*$*$$")
