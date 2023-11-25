# Importing Libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds

# Disable progress bar for TensorFlow Datasets
tfds.disable_progress_bar()

# Load and split the "cats_vs_dogs" dataset
(raw_train, raw_validation, raw_test), metadata = tfds.load(
    name='cats_vs_dogs',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)

# Map label integers to string names
get_label_name = metadata.features['label'].int2str

# Visualize two images from the training set
for image, label in raw_train.take(2):
    plt.figure()
    plt.imshow(image)
    plt.title(get_label_name(label))

# Define image size for preprocessing
IMG_SIZE = 160

# Define a function to preprocess examples
def format_example(image, label):
    image = tf.cast(image, tf.float32)
    image = (image/127.5)-1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label

# Apply the preprocessing function to the datasets
train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)

# Visualize preprocessed images
for image, label in train.take(2):
    plt.figure()
    plt.imshow(image)
    plt.title(get_label_name(label))

# Set batch size and shuffle buffer size
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

# Create batches of data
train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)

# Display original and new shapes of images
for img, label in raw_train.take(2):
    print('Original shape:', img.shape)
for img, label in train.take(2):
    print('New shape', img.shape)

# Importing the model:

# Define image shape
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

# Import MobileNetV2 model as the base model
base_model = keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                            include_top=False,
                                            weights='imagenet')
#base_model.summary()

# Display the shape of feature batch
for image, _ in train_batches.take(1):
    pass
feature_batch = base_model(image)
print(feature_batch.shape)

# Set the base model as non-trainable
base_model.trainable = False
#base_model.summary()

# Define global average pooling and dense prediction layers
global_average_layer = keras.layers.GlobalAveragePooling2D()

prediction_layer = keras.layers.Dense(1)

# Build the final model using Sequential API
model = keras.Sequential()
model.add(base_model)
model.add(global_average_layer)
model.add(prediction_layer)

#model.summary()

# Compile the model
base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr = base_learning_rate),
              loss = keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Define initial epochs and validation steps
initial_epochs = 3
validation_steps = 20

# Evaluate the model on validation data
loss0, accuraxy0 = model.evaluate(validation_batches, steps = validation_steps)

# Train the model
history = model.fit(train_batches,
                    epochs=initial_epochs,
                    validation_data=validation_batches)

# Access accuracy from the training history
acc = history.history['accuracy']
#print(acc)

# Save the trained model
model.save('dogs_vs_cats.h5')

# Load the saved model
new_model = tf.keras.models.load_model('dogs_vs_cats.h5')

