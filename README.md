<h1 align="center">Deep Learning Exploration with TensorFlow</h1>
<br>
Welcome to my repository dedicated to delving into the captivating realm of Deep Learning using TensorFlow!

<br>

<h2 align="center">🌅 Journey Highlights 🌅</h2>
My initiation into the world of Deep Learning started with the enlightening 6.S191 MIT course by Alexander Amini. Building on this foundation, I expanded my knowledge by following the instructive courses curated by Josh Stamer. The hands-on experience in the practical application of Deep Learning with TensorFlow was facilitated by Aladdin Persson's tutorials, culminating in being certified TensorFlow Developer by Google.
I express my gratitude to these educators for providing excellent and free resources to the community.

<br>

---

Before diving into the projects, you'll find a comprehensive list of abbreviations and terms
<br>
<details>
  <h2 align="center"> 📚 Definitions 📚 </h2>
  
  <summary> 📚 Definitions 📚</summary> 
<p>
  
**DL (Deep-Learning):** A subset of machine learning that involves training artificial neural networks on vast amounts of data to make intelligent decisions without explicit programming.

**DNN (Deep Neural Networks):** A class of neural networks with multiple layers (deep architecture) between the input and output layers, enabling the model to learn complex hierarchical representations.

**CNN (Convultional Neural Network):** A type of deep neural network specifically designed for processing grid-like data, such as images, using convolutional layers to automatically and adaptively learn spatial hierarchies of features.

**FNN (Feedforward Neural Network):** A basic neural network architecture where information travels in one direction, from the input layer through hidden layers to the output layer, without forming cycles.

**RNN (Recurrent Neural Network):** A type of neural network designed for sequence tasks, where connections between nodes form directed cycles, allowing information persistence and handling sequential dependencies

**NLP (Natural Language Processing):** A field of artificial intelligence that focuses on the interaction between computers and human language, enabling machines to understand, interpret, and generate human-like text.

**LSTM (Long Short-Term Memory):** A type of recurrent neural network architecture designed to capture and remember long-term dependencies in sequential data, mitigating the vanishing gradient problem often encountered in standard RNNs.
</p>
  <br>
</details>

<br>

<h2 align="center">🔎 Repository Overview 🔍</h2>
<br>
This repository is a testament to my exploration and experimentation within the domain of Deep Learning. It is divided into two primary sections:

<br>

<h3 align="center">Deep-Learning Applications</h3>
<br>
Each project is accompanied by a brief overview, outlining the goals and methodologies employed. I approached these projects with the intent to learn new techniques, push the boundaries of TensorFlow, to strengthen my skills.

<br>

<details>
  <h2 align="center"> 🎭 LSTM NLP RNN Shakespeare Play Generator & User interaction 🎭 </h2>
  
  <summary> 🎭 LSTM NLP RNN Shakespeare Play Generator & User interaction 🎭 </summary> 

  <p>
The purpose of this code is to train a character-level LSTM neural network on a dataset containing Shakespearean text. 
    
The trained model is designed to learn the patterns and structures inherent in the language of Shakespeare. Subsequently, the model can generate new text based on a user-provided starting string. This demonstrates the use of recurrent neural networks for creative text generation, showcasing the network's ability to capture and reproduce the linguistic style of a specific author or domain. The code engages users by allowing them to input a seed string and witness the model's generation of coherent and contextually relevant text in the style of Shakespeare.
  </p>
  <br>
</details>

<br>

<details>
  <h2 align="center">👹 NLP Sarcasm Classifier 👹</h2>
  
  <summary>👹 NLP Sarcasm Classifier 👹</summary> 
  <p>
This Python script constructs and trains a sarcasm classifier using TensorFlow and Keras. 
    
The process involves importing essential libraries (json, tensorflow, numpy, and relevant modules from tensorflow.keras), loading and preprocessing the sarcasm dataset, tokenizing and padding text data, building a sequential model with layers for word embeddings, dropout, global average pooling, and dense classification. Early stopping is implemented with a patience of 5 epochs to monitor validation loss. The model is compiled using Adam optimizer and binary cross-entropy loss, trained for 50 epochs with validation data, and the trained model is saved as "mymodel.h5". The script is designed to be executed as the main program, invoking the solution_model function for model creation and saving. 

This script aims to create a straightforward text classification model for sarcasm detection, incorporating neural network elements and measures to enhance its effectiveness. The accuracy on the validation dataset is +0.95.
  </p>
  <br>
</details>

<br>

<details>
  <h2 align="center"> 🪻 DNN Iris flower Prediction & User Interaction 🪻</h2>
  
  <summary> 🪻 DNN Iris flower Prediction & User Interaction 🪻 </summary> 

  <p>
Use the trained classifier to make predictions based on the user's input and print the predicted class and its probability.

In summary, this code defines, trains, evaluates, and uses a Deep Neural Network classifier to predict the species of an Iris flower based on user-inputted features. The dataset used is the famous Iris dataset containing features such as sepal length, sepal width, petal length, and petal width. The user can interactively input feature values for prediction.
  </p>
  <br>
</details>

<br>

<details>
  <h2 align="center">9️⃣ FNN MNIST Prediction & User interaction 9️⃣</h2>
  
  <summary> 9️⃣ FNN MNIST Prediction & User interaction 9️⃣ </summary> 

  <p>
This script utilizes TensorFlow and Keras to implement a Feedforward Neural Network for classifying Fashion MNIST images into 10 different categories. 
    
The dataset is loaded, preprocessed by scaling pixel values, and then used to build a sequential custom model with one input layer, one hidden layer with 128 neurons and ReLU activation, and one output layer with softmax activation. The model is compiled with the Adam optimizer and sparse categorical crossentropy loss. Additionally, the code defines functions for predicting and displaying the results of the model on a chosen image from the test set.

The primary goal is to showcase the process of building, training, and interacting with a neural network for image classification. The chosen dataset, Fashion MNIST, consists of grayscale clothing images, and the script demonstrates the model's predictions on a user-selected test image.
  </p>
  <br>
</details>

<br>

<details>
  <h2 align="center"> 🔁 Reinforcement Learning: Q Learning 🔁</h2>
  
  <summary> 🔁 Reinforcement Learning: Q Learning 🔁 </summary> 

  <p>
The goal of the code is to implement a Q-learning algorithm to train an agent in the FrozenLake environment, a classic problem in reinforcement learning.
    
The code initializes a Q-table to store the learned values for state-action pairs and iteratively updates these values based on the agent's interactions with the environment. The training process involves a balance between exploration and exploitation, where the agent chooses actions with a certain probability of exploration. The Q-values are updated using the Q-learning formula, taking into account the rewards received and the maximum Q-value for the next state. The training loop runs for a specified number of episodes, and the final learned Q-values are printed along with the average reward obtained during training. The ultimate objective is for the agent to learn an optimal policy for navigating the FrozenLake environment and achieving the highest cumulative reward.
  </p>
  <br>
</details>

<br>

<details>
  <h2 align="center">🌪️ CNN Hurrican Classifier 🌪️</h2>
  
  <summary>🌪️ CNN Hurrican Classifier 🌪️ </summary> 

  <p>
This Python script employs a Convolutional Neural Network (CNN) to classify post-hurricane satellite images into "damage" and "no_damage" categories. 
    
It utilizes the "satellite-images-of-hurricane-damage" dataset, sourced from Texas after Hurricane Harvey. The script includes functions for dataset handling, image normalization, and model creation using TensorFlow. The model is trained for 30 epochs, achieving a validation accuracy of +0.92, and is saved as "mymodel.h5" for future use. 

The overall goal is to demonstrate the process of preparing a dataset, constructing a CNN model, training, and saving it.
</p>
<br>
</details>

<br>

<details>
  <h2 align="center"> 🦠 Tranfer-Learning Skin Cancer Classifier 🦠 </h2>
  
  <summary> 🦠 Tranfer-Learning Skin Cancer Classifier 🦠 </summary> 

  <p>
This project aims to develop a binary image classification model using a pre-trained EfficientNet from TensorFlow Hub. 
    
The goal is to achieve high accuracy in distinguishing between two classes in a dataset of images. The approach involves implementing data augmentation techniques for improved model generalization and training. The model is evaluated on both validation and test datasets, with performance metrics such as accuracy, precision, recall, and the ROC curve used to assess its effectiveness. The use of transfer learning with a powerful pre-trained neural network enables efficient feature extraction and classification for image recognition tasks. The project leverages TensorFlow and related libraries for seamless model development, training, and evaluation.
  </p>
  <br>
</details>

<br>

<details>
  <h2 align="center"> 🐶 Pre-Trained Model Cat VS Dog Classifier 🐱 </h2>
  
  <summary> 🐶 Pre-Trained Model Cat VS Dog Classifier 🐱 </summary> 

  <p>
    The code loads the "cats_vs_dogs" dataset, preprocesses the images, and fine-tunes the MobileNetV2 model for a binary classification task (cats vs. dogs). It trains the model, evaluates its performance, and saves the trained model for later use. The primary goal is to demonstrate the process of using a pre-trained neural network for image classification and adapting it to a specific task. It deliver an accuracy of +0.98 on validation dataset.
  </p>
  <br>
</details>

<br>

<details>
  <h2 align="center"> 🦆 CNN Cifar-10 🐴 </h2>
  
  <summary> 🦆 CNN Cifar-10 🐴 </summary> 

  <p>
This code demonstrates the use of data augmentation to artificially increase the diversity of the training dataset, enhancing the model's ability to generalize to unseen data. The CNN model is designed to classify images from the CIFAR-10 dataset into one of the ten specified classes. The training process involves both the original and augmented images.
  </p>
  <br>
</details>

<br>

---

<h2 align="center"> 💫 Specifics Custom Technics from TensoFlow 💫 </h2>

<br>
<details>
  <h2 align="center"> Custom Callbacks MNIST </h2>
  
  <summary href="https://github.com/trystan-geoffre/Deep-Learning-TensorFlow/blob/master/Custom_Callbacks-MNIST.py"> Custom Callbacks MNIST </summary> 

  <p>
The code employs three callbacks during the training process. The ModelCheckpoint Callback saves model weights at specified intervals, the LearningRateScheduler Callback dynamically adjusts the learning rate, and the CustomCallback Callback stops training if the accuracy surpasses a predefined threshold of 90%. These callbacks enhance training control and efficiency, ensuring periodic weight saving, adaptive learning rates, and the ability to halt training based on a specific criterion.
  </p>
  <br>
</details>

<br>

<details>
  <h2 align="center"> Custom Layers ResNet </h2>
  
  <summary >Custom Layers ResNet </summary> 

  <p>
The custom layers, CNNBlock and ResBlock, play a key role in constructing a ResNet-like model for MNIST digit classification. They enhance expressiveness by incorporating convolutional blocks and facilitating the creation of residual blocks, enabling efficient feature learning and mitigating challenges in training deep neural networks.

<a href=""> Code Link</a>
  </p>
  <br>
</details>

<br>

<details>
  <h2 align="center"> Custom Training Loop MNIST </h2>
  
  <summary>Custom Training Loop MNIST </summary> 

  <p>
The custom training loop in the code offers greater flexibility and control over the training process compared to the standard model.fit() method. It allows explicit definition of operations such as model updates, loss calculations, and metric tracking, providing transparency and adaptability during training.
  </p>
  <br>
</details>

<br>

<details>
  <h2 align="center"> Custom Model Fit MNIST </h2>
  
  <summary>Custom Model Fit MNIST </summary> 

  <p>
The custom model fit in this code provides a specialized training loop for a convolutional neural network (CNN) on the MNIST dataset. It allows fine-grained control over training and evaluation, incorporating specific metrics like sparse categorical accuracy and utilizing an Adam optimizer with sparse categorical cross-entropy loss. This customization enhances adaptability and transparency in the training process.
  </p>
  <br>
</details>

<br>

