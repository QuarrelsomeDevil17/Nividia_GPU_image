#implement the RNN model
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. Load the IMDB dataset
# num_words=10000 means we only consider the top 10,000 most frequent words
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

# 2. Preprocess the Data
# Set the maximum sequence length (truncate or pad sequences to this length)
max_seq_length = 500

# Pad sequences to ensure they are all the same length
x_train = pad_sequences(x_train, maxlen=max_seq_length)
x_test = pad_sequences(x_test, maxlen=max_seq_length)

# 3. Build the RNN Model
model = Sequential()

# Embedding layer to learn word representations
# input_dim=10000 -> Vocabulary size
# output_dim=32 -> Dimensionality of the embedding vectors
# input_length=max_seq_length -> Length of input sequences
model.add(Embedding(input_dim=10000, output_dim=32, input_length=max_seq_length))

# SimpleRNN layer with 100 units
model.add(SimpleRNN(100, activation='relu'))

# Dense layer with a single neuron and sigmoid activation for binary classification
model.add(Dense(1, activation='sigmoid'))

# Compile the model with binary crossentropy loss and the Adam optimizer
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 4. Train the Model
# Use batch_size of 64 and train for 5 epochs
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))

# 5. Evaluate the Model
# Evaluate the model on the test data to see how well it performs
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# 6. Make Predictions
# Predict the sentiment of the first review in the test set
prediction = model.predict(x_test[:1])
print(f'Predicted sentiment: {"Positive" if prediction[0][0] > 0.5 else "Negative"}')