# This Python script builds and trains a classifier for a sarcasm dataset
# using TensorFlow and Keras.

# Import the libraries
import json
import tensorflow as tf
import numpy as np
import urllib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def solution_model():
    # Import the dataset
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/sarcasm.json'
    urllib.request.urlretrieve(url, 'sarcasm.json')

    # Load the data from JSON file
    with open('../sarcasm.json', 'r') as f:
        datastore = json.load(f)

    # Extract sentences and labels
    sentences = []
    labels = []
    for item in datastore:
        sentences.append(item['headline'])
        labels.append(item['is_sarcastic'])

    vocab_size = 10000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_size = 20000

    # Split the data into training and validation sets
    train_sentences = sentences[:training_size]
    train_labels = labels[:training_size]
    val_sentences = sentences[training_size:]
    val_labels = labels[training_size:]

    # Tokenization and padding
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(train_sentences)
    train_sequences = tokenizer.texts_to_sequences(train_sentences)
    train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    val_sequences = tokenizer.texts_to_sequences(val_sentences)
    val_padded = pad_sequences(val_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    train_labels = np.array(train_labels)
    val_labels = np.array(val_labels)

    # Create the model
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
        tf.keras.layers.Dropout(0.5),  # Dropout layer to reduce overfitting
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Code for early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Compile and fit the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_padded, train_labels, validation_data=(val_padded, val_labels),
              epochs=50, verbose=1,  callbacks=[early_stopping])

    return model

# Run and save the model
if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")

