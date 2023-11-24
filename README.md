
<details>
  <h2 align="center">🌪️ CNN Hurrican Classifier 🌪️</h2>
  
  <summary>🌪️ CNN Hurrican Classifier 🌪️ </summary> 

  <p>
    This Python script focuses on creating and training a Convolutional Neural Network (CNN) to classify images into two classes ("damage" and "no_damage") using the "satellite-images-of-hurricane-damage" dataset. Here's a breakdown of the script:

About the Dataset:
The dataset is sourced from Texas after Hurricane Harvey, containing satellite images categorized into "damage" and "no_damage" groups.
Original Source: IEEE DataPort - Detecting Damaged Buildings Post-Hurricane.
Libraries:
The script begins by importing necessary libraries, including urllib for handling URLs and zipfile for extracting compressed files.
TensorFlow is imported as tf.
Functions:
download_and_extract_data(): Downloads and extracts the dataset from a specified URL, storing it in the current directory.
preprocess(image, label): Normalizes the images by scaling pixel values between 0 and 1.
Solution Model Function (solution_model()):
Calls download_and_extract_data() to prepare the dataset.
Defines constants for image size (IMG_SIZE) and batch size (BATCH_SIZE).
Loads training and validation datasets from the "train/" and "validation/" directories, respectively. It resizes the images and forms batches using TensorFlow's image_dataset_from_directory function.
Normalizes the datasets using the preprocess function.
Constructs a CNN model using TensorFlow's Sequential API with convolutional and pooling layers, flattening layer, and dense layers. The final layer uses the sigmoid activation function for binary classification.
Compiles and trains the model using the Adam optimizer and binary cross-entropy loss for 30 epochs.
Running and Saving the Model (if __name__ == '__main__':):
Calls solution_model() to create and train the model.
Saves the trained model as "mymodel.h5".
Overall, the script demonstrates the process of preparing a dataset, creating a CNN model, training the model, and saving it for future use. The goal is to classify satellite images into "damage" and "no_damage" categories.
</p>
<details>

