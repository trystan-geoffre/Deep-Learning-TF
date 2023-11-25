# Import Libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds

# Load MNIST Dataset
(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)

# Normalize Images and Configure Dataset
def normalize_img(image, label):
    return tf.cast(image, tf.float32)/255.0, label

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 128

ds_train = ds_train.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(BATCH_SIZE)
ds_train = ds_train.prefetch(AUTOTUNE)

# Define Convolutional Neural Network (CNN) Model
model = keras.Sequential(
    [
        keras.Input((28,28,1)),
        layers.Conv2D(32, 3, activation='relu'),
        layers.Flatten(),
        layers.Dense(10)
    ]
)

# Configure Callbacks
save_callback = keras.callbacks.ModelCheckpoint(
    'checkpoint/',
    save_weights_only=True,
    monitor='accuracy',
    save_best_only=False,

)

def scheduler(epoch, lr):
    if epoch<2:
        return lr
    else:
        return lr*0.99

lr_scheduler = keras.callbacks.LearningRateScheduler(scheduler, verbose=1)

# Defining limit to Callbacks
class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy') > 0.90:
            print('\n Accuracy over 90%, quitting training \n')
            self.model.stop_training = True


# Compile and Train Model
model.compile(
    optimizer=keras.optimizers.legacy.Adam(0.01),
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
              )

# Model fitting
model.fit(
    ds_train,
    epochs=10,
    verbose=1,
    callbacks = [save_callback,lr_scheduler, CustomCallback()],
)

