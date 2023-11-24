Motivation & Goals:


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
<br>
</details>

<br>

<details>
  <h2 align="center">👹 NLP Sarcasm Classifier 👹</h2>
  
  <summary>👹 NLP Sarcasm Classifier 👹</summary> 
  <p>
    This Python script builds and trains a classifier for a sarcasm dataset using TensorFlow and Keras. Here's an explanation of each part:

Import Libraries:
Import necessary libraries including json for working with JSON files, tensorflow for building and training the model, numpy for numerical operations, and relevant modules from tensorflow.keras for text preprocessing.
Load and Preprocess Data:
Download the sarcasm dataset from a given URL and load it from the JSON file (sarcasm.json).
Extract sentences and labels from the dataset.
Tokenization and Padding:
Tokenize the text data and pad sequences to ensure uniform length for model input.
Model Architecture:
Build a sequential model using Keras.
The model consists of an embedding layer for word embeddings, a dropout layer to prevent overfitting, a global average pooling layer for dimensionality reduction, and a dense layer with a sigmoid activation function for binary classification.
Early Stopping:
Implement early stopping with a patience of 5 epochs to monitor validation loss and restore the best weights when there is no improvement.
Compile and Train:
Compile the model using the Adam optimizer and binary cross-entropy loss.
Train the model on the training data with validation data for 50 epochs, using the early stopping callback to prevent overfitting.
Save the Model:
Save the trained model to a file named "mymodel.h5".
Main Execution:
Check if the script is being run as the main program (__name__ == '__main__').
If yes, execute the solution_model function and save the resulting model.
This script aims to create a simple text classification model for sarcasm detection using a neural network architecture with embedding layers, dropout for regularization, and early stopping to prevent overfitting.
  </p>
  <br>
</details>

<br>

<details>
  <h2 align="center"> CNN Cifar10 </h2>
  
  <summary> CNN Cifar10 </summary> 

  <p>
This code demonstrates the use of data augmentation to artificially increase the diversity of the training dataset, enhancing the model's ability to generalize to unseen data. The CNN model is designed to classify images from the CIFAR-10 dataset into one of the ten specified classes. The training process involves both the original and augmented images.
  </p>
  <br>
</details>

<br>

<details>
  <h2 align="center"> DNN Iris flower Prediction </h2>
  
  <summary> DNN Iris flower Prediction </summary> 

  <p>
Use the trained classifier to make predictions based on the user's input and print the predicted class and its probability.

In summary, this code defines, trains, evaluates, and uses a Deep Neural Network classifier to predict the species of an Iris flower based on user-inputted features. The dataset used is the famous Iris dataset containing features such as sepal length, sepal width, petal length, and petal width. The user can interactively input feature values for prediction.
  </p>
  <br>
</details>

<br>

<details>
  <h2 align="center">🌪️ CNN Hurrican Classifier 🌪️</h2>
  
  <summary>🌪️ CNN Hurrican Classifier 🌪️ </summary> 

  <p>

  </p>
  <br>
</details>

<br>

<details>
  <h2 align="center">🌪️ CNN Hurrican Classifier 🌪️</h2>
  
  <summary>🌪️ CNN Hurrican Classifier 🌪️ </summary> 

  <p>

  </p>
  <br>
</details>

<br>

<details>
  <h2 align="center">🌪️ CNN Hurrican Classifier 🌪️</h2>
  
  <summary>🌪️ CNN Hurrican Classifier 🌪️ </summary> 

  <p>

  </p>
  <br>
</details>
