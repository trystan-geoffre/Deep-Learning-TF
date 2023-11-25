# Importing Libraries
import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

# Loading Fashion MNIST Dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels)= fashion_mnist.load_data()

# Class Names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Data Preprocessing
train_images = train_images/255.0
test_images = test_images/255.0

# Building the Model
model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28,28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compiling the Model
opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)
model.compile(optimizer = opt,
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

# Training the Model
model.fit(train_images, train_labels, epochs=8)

# Visualization :
# These lines of code are commented out and are used
# for evaluating the model's accuracy, making predictions,
# and visualizing an image from the training set
#test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)
#Print('Test accuracy:', test_acc)

#predictions = model.predict([test_images])
#print(class_names[np.argmax(predictions[0])])
#plt.figure()
#plt.imshow(train_images[0])
#plt.colorbar()
#plt.grid(False)
#plt.show()

COLOR = 'white'
plt.rcParams['text.color']=COLOR
plt.rcParams['axes.labelcolor'] = COLOR

# Takes an image, correct label, and predicts the class using the trained model
def predict(model, image, correct_label):
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    prediction = model.predict(np.array([image]))
    predicted_class = class_names[np.argmax(prediction)]

    show_image(image, class_names[correct_label], predicted_class)

# Displays the image along with the expected and predicted labels.
def show_image(img, label, guess):
    plt.figure()
    plt.imshow(img)
    print('Expected:'+label)
    print('Guess:' + guess)
    plt.colorbar()
    plt.grid(False)
    plt.show()

def get_number():
    while True:
        num = input('Pick a number:')
        if num.isdigit():
            num = int(num)
            if 0 <=num <= 1000:
                return int(num)
            else:
                print('Try again...')

# User Interaction
num = get_number()
image = test_images[num]
label = test_labels[num]
predict(model, image, label)
