
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# Check version of TensorFlow
print(tf.__version__)

# Get data
(train_images, train_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()

# Normalize images
train_images, test_images = train_images / 255.0, test_images / 255.0

# Build the model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),  # Flatten the 28x28 input images
    layers.Dense(128, activation='relu'),   # Fully connected layer with 128 units
    layers.Dense(10, activation='softmax')  # Fully connected layer with 10 units for 10 classes
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Print model summary
model.summary()

# Fit the model
model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_accuracy}")

# Save the model
#model.save("test_image_model2.h5")




