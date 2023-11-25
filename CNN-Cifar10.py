# Import the libraries
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from keras import layers, models
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

# Load and Normalize Dataset
(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# Define Class Names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'frog', 'dog', 'horse', 'ship', 'truck']

# DATA AUGMENTATION
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Generate augmented images
augmented_images = []

test_img = train_images[14]
img = image.img_to_array(test_img)
img = img.reshape((1,) + img.shape)

i = 0

# Generate Augmented Images
for batch in datagen.flow(img, save_prefix='test', save_format='jpeg'):
    augmented_images.append(image.img_to_array(batch[0]))
    i = i + 1
    if i > 4:
        break
#To show 4 augmented images
#plt.show()


# Convert augmented_images to numpy array
augmented_images = np.array(augmented_images)

# Create the model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Flatten())
model.add(layers.Dropout(0.2))

# Hidden Layer
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.2))

# Output Layer
model.add(layers.Dense(10))
model.summary()

# Compile and fit the Model
model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Fit the model using augmented images
model.fit(datagen.flow(train_images, train_labels, batch_size=32),  # Use datagen.flow for augmented training data
          epochs=20,
          validation_data=(test_images, test_labels))

# Testing Model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(test_acc)

