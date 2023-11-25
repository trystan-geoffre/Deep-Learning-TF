# Download and Load Text Data
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import sequence
import os
import numpy as np

path_to_file = keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
#print('length f text:{} characters'.format(len(text)))
#print(text[:250])

# Text Processing
# Here, it creates a vocabulary of unique characters
# and builds dictionaries (char2idx and idx2char)
# for mapping characters to indices and vice versa.
vocab = sorted(set(text))
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

# Text Encoding
def text_to_int(text):
    return np.array([char2idx[c] for c in text])

text_as_int = text_to_int(text)

#print('text:', text[:13])
#print('Encoded:', text_to_int(text[:13]))

def int_to_text(ints):
    try:
        ints = ints.numpy()
    except:
        pass
    return ''.join(idx2char[ints])
#print(int_to_text(text_as_int[:13]))

# Create Sequences
seq_length = 100
exemples_per_epochs = len(text)//(seq_length+1)

char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

# Build Dataset
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)

# Model Building
BATCH_SIZE = 64
VOCAB_SIZE = len(vocab)
EMBEDDING_DIM = 256
RNN_UNITS = 1024

BUFFER_SIZE = 10000

data = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = keras.Sequential([
        keras.layers.Embedding(vocab_size, embedding_dim,
                               batch_input_shape=[batch_size, None]),
        keras.layers.LSTM(rnn_units,
                          return_sequences=True,
                          stateful=True,
                          recurrent_initializer='glorot_uniform'),
        keras.layers.Dense(vocab_size)
    ])
    return model

model = build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, BATCH_SIZE)
#model.summary()

for input_exemple_batch, target_exemple_batch in data.take(1):
    example_batch_predictions = model(input_exemple_batch)
    #print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

#print(len(example_batch_predictions))
#print(example_batch_predictions)

pred = example_batch_predictions[0]
#print(len(pred))
#print(pred)

sampled_indices = tf.random.categorical(pred, num_samples=1)
sampled_indices = np.reshape(sampled_indices, (1,-1))[0]
predicted_chars = int_to_text(sampled_indices)

#print(predicted_chars)


def loss(labels, logits):
    return keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

model.compile(optimizer='adam', loss=loss)

# Model Training
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt_{epoch}')

checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True
)

history = model.fit(data, epochs=40, callbacks=[checkpoint_callback])

# Text Generation
model = build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, batch_size=1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))

#checkpoint_num = 10
#model.load_weights(tf.train.load_checkpoint("./training_checkpoints/ckpt_" + str(checkpoint_num)))
#model.build(tf.TensorShape([1,None]))

# Function to generate text
def generate_text(model, start_string):
    num_generate = 800

    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []

    temperature = 1.0

    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)

        predictions = predictions/temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append((idx2char[predicted_id]))
    return  (start_string + ''.join(text_generated))

# User Interaction
inp = input('Type a starting string: ')
print(generate_text(model, inp))



