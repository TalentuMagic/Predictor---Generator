import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.optimizers import Adam
import pandas as pd

# Assuming you have loaded your dataset into a pandas DataFrame df
df = pd.read_csv("dataSet.csv")
# Split the data into input (X) and output (y)
X = df.iloc[:, :17]  # columns 1-17
y = df.iloc[:, 19]   # column 19

# Normalize X
X = X / np.max(X)

# Reshape X to be suitable for LSTM (samples, timesteps, features)
X = np.reshape(X.values, (X.shape[0], 1, X.shape[1]))

# Define the LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(1))

# Compile the model
model.compile(loss='mean_squared_error',
              optimizer=Adam(), metrics=['accuracy'])

# Fit the model
model.fit(X, y, epochs=100, batch_size=32, verbose=2)

# To make predictions


def predict_next_label(input_sequence):
    input_sequence = np.reshape(input_sequence, (1, 1, len(input_sequence)))
    predicted_label = model.predict(input_sequence)
    return predicted_label
