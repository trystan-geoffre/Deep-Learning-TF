# This Python script focuses on creating and training a Convolutional Neural Network (CNN)
# to classify images into two classes ("damage" and "no_damage")
# The dataset consists of satellite images from Texas after Hurricane Harvey

# Source:
# https://ieee-dataport.org/open-access/detecting-damaged-buildings-post-hurricane-satellite-imagery-based-customized



# Import the libraries
import urllib
import zipfile

import tensorflow as tf

# This function downloads and extracts the dataset to the directory
def download_and_extract_data():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/certificate/satellitehurricaneimages.zip'
    urllib.request.urlretrieve(url, 'satellitehurricaneimages.zip')
    with zipfile.ZipFile('../satellitehurricaneimages.zip', 'r') as zip_ref:
        zip_ref.extractall()

# This function normalizes the images.
def preprocess(image, label):
    image = image / 255.0
    return image, label



def solution_model():
    # Downloads and extracts the dataset to the directory that
    download_and_extract_data()

    IMG_SIZE = (128, 128)
    BATCH_SIZE = 64

    # Load the data into train and validation datasets
    # Resize them into the specified image size and splits them into batches
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory='train/',
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory='validation/',
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE)

    # Normalize the train and validation datasets using the preprocess() function
    train_ds = train_ds.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(
        tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Define the model
    model = tf.keras.models.Sequential([

        # Being careful with the shape of the input layer and of the output
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])

    # Compile and train the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy']
    )

    model.fit(train_ds,
              validation_data=val_ds,
              epochs=30
    )

    return model


# Run and save the model
if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")
