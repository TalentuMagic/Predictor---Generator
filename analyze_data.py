from collections import Counter
from math import sqrt
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
from datetime import datetime
import asyncio
import numpy as np

# load the dataset's metadata
dataset = pd.read_csv("dataSet.csv")

vectorizer = CountVectorizer()

X = np.array(dataset.iloc[:, :17])
# Join all string elements together
string_elements = ''.join([elem for elem in X[1] if isinstance(elem, str)])
# Count the occurrences of each character
features = Counter(string_elements)
print(features)

raise SystemExit()

# taking the number of labels available for both input and output data
num_labels = X.shape[0]

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X, y, test_size=0.2)

# print(X2_train.shape, y2_train.shape)

# define the Neural Network model
model = tf.keras.Sequential()
# start with num_labels neurons (in this case 2142), with ReLU activation and a L2 regularizer to prevent overfitting
model.add(tf.keras.layers.Dense(num_labels, activation='relu', input_shape=(
    1,), kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(tf.keras.layers.Dropout(0.2))
# the next two layers consists of a divisor of the num_labels
model.add(tf.keras.layers.Dense(int(num_labels/1.5), activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(int(num_labels/2), activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
# the output layer has the num_labels (in this case 56)
model.add(tf.keras.layers.Dense(1, activation='softmax'))

# compiling the model
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

num_epochs = 8  # Start with a moderate number of epochs and adjust if needed
batch_size = 64  # Adjust based on available memory and model complexity

# each time the model gets updated, the checkpointer updated the .hdf5 file (Hierarchical Data Format)
checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=f'./analysed_Data_line{line}.h5',
                                                  verbose=1, save_best_only=True)
# fit the model according to the neural network and the configurations done above
model.fit(X2_train, y2_train, batch_size=batch_size, epochs=num_epochs,
          validation_data=(X2_test, y2_test), callbacks=[checkpointer], verbose='1')

predict = model.predict(input_line).astype('int32')
prediction_class = labelEncoder.inverse_transform(predict)
print(prediction_class)
