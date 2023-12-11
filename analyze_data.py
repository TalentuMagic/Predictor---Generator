from math import sqrt
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from tqdm import tqdm
from datetime import datetime
import asyncio
import numpy as np

# load the dataset's metadata
dataset = pd.read_csv("dataSet.csv")
labelEncoder = LabelEncoder()


def model_line1(dataset: pd.DataFrame):
    uniques = dataset.iloc[:, [17, 1, 18, 3]].drop_duplicates()
    print(uniques)
    X = np.array(uniques.iloc[:, [0, 1]])
    y = np.array(uniques.iloc[:, [2, 3]])

    # print(X, X.shape)
    # print(y, y.shape)

    # split data into train and test arrays
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # define the model's layers
    input_layer = tf.keras.layers.Input(shape=(2,))
    hidden_layer = tf.keras.layers.Dense(64, activation='relu')(input_layer)
    output_layer1 = tf.keras.layers.Dense(1, name='label_output')(
        hidden_layer)  # output layer for the label prediction
    output_layer2 = tf.keras.layers.Dense(1, name='class_output')(
        hidden_layer)  # output layer for the class prediction

    # define the model
    model = tf.keras.models.Model(inputs=input_layer, outputs=[
        output_layer1, output_layer2])

    # compile the model
    model.compile(optimizer='adam',
                  loss={'label_output': 'mean_squared_error',
                        'class_output': 'mean_squared_error'},
                  metrics={'label_output': ['mae'], 'class_output': ['accuracy']})
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'best_model.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)
    history = model.fit(X_train, {'label_output': y_train[:, 0], 'class_output': y_train[:, 1]},
                        validation_data=(
                            X_val, {'label_output': y_val[:, 0], 'class_output': y_val[:, 1]}),
                        epochs=16, batch_size=32, callbacks=[checkpoint])
    loss, label_loss, class_loss, label_mae, class_accuracy = model.evaluate(
        X_val, {'label_output': y_val[:, 0], 'class_output': y_val[:, 1]})

    predictions = model.predict(np.array([[145, 4]]))
    predictions = [np.round(pred) for pred in predictions]
    print(predictions)


def model2_line1(dataset: pd.DataFrame):
    uniques = dataset.iloc[:, [17, 1, 18, 3]].drop_duplicates()
    X = np.array(uniques.iloc[:, [0, 1]])
    y = np.array(uniques.iloc[:, [2, 3]])

    label_encoders = [LabelEncoder() for _ in range(y.shape[1])]
    # Reshape y to 1D and fit and transform the label encoder
    y = labelEncoder.fit_transform(y.reshape(-1)).reshape(y.shape)
    # split data into train and test arrays
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42)

    num_classes1 = len(np.unique(y_train[:, 0]))
    num_classes2 = len(np.unique(y_train[:, 1]))

    # define the model's layers
    input_layer = tf.keras.layers.Input(shape=(2,))
    hidden_layer = tf.keras.layers.Dense(64, activation='relu')(input_layer)
    output_layer1 = tf.keras.layers.Dense(
        num_classes1, activation='softmax', name='label_output')(hidden_layer)
    output_layer2 = tf.keras.layers.Dense(
        num_classes2, activation='softmax', name='class_output')(hidden_layer)

    # define the model
    model = tf.keras.models.Model(inputs=input_layer, outputs=[
                                  output_layer1, output_layer2])

    # compile the model
    model.compile(optimizer='adam',
                  loss={'label_output': 'categorical_crossentropy',
                        'class_output': 'categorical_crossentropy'},
                  metrics={'label_output': ['accuracy'], 'class_output': ['accuracy']})

    # convert y to categorical
    y_train_cat = [tf.keras.utils.to_categorical(
        y, num_classes=num_classes1) for y in y_train.T]
    y_val_cat = [tf.keras.utils.to_categorical(
        y, num_classes=num_classes2) for y in y_val.T]

    # define the checkpoint
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'best_model.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)

    # fit the model
    history = model.fit(X_train, {'label_output': y_train_cat[0], 'class_output': y_train_cat[1]},
                        validation_data=(
                            X_val, {'label_output': y_val_cat[0], 'class_output': y_val_cat[1]}),
                        epochs=16, batch_size=32, callbacks=[checkpoint])

    loss, label_loss, class_loss, label_accuracy, class_accuracy = model.evaluate(
        X_val, {'label_output': y_val_cat[0], 'class_output': y_val_cat[1]})

    # predict the class with the highest probability
    predictions = model.predict(X_val)
    predicted_classes = [np.argmax(pred, axis=-1) for pred in predictions]
    print(predicted_classes)


model_line1(dataset=dataset)
