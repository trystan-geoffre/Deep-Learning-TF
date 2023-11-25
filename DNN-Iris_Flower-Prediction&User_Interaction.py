# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

#  Load Iris Dataset
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

train_path = tf.keras.utils.get_file(
    'iris_trainning.csv', 'https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv')
test_path = tf.keras.utils.get_file(
    'iris_test.csv', 'https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv')

train = pd.read_csv(train_path, names = CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names = CSV_COLUMN_NAMES, header=0)

# Prepare Data
train_y = train.pop('Species')
test_y = test.pop('Species')

# Define Input Function
def input_fn(features, labels, training=True, batch_size=256):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    if training:
        dataset = dataset.shuffle(1000).repeat()
    return dataset.batch(batch_size)

# Feature Columns
my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))

# Define the Classifier
classifier = tf.estimator.DNNClassifier(
    feature_columns = my_feature_columns,
    hidden_units=[30,10],
    n_classes=3
)

# Train the Classifier
classifier.train(
    input_fn = lambda:input_fn(train, train_y, training=True),
    steps=5000
)

# Evaluate the Model
eval_result = classifier.evaluate(input_fn = lambda: input_fn(test, test_y, training=False))
#print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

# User Input for Prediction
def input_fn(features, batch_size = 256):
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

features = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
predict={}

print('Please type numeric values in this format: x.x')
for feature in features:
    valid = True
    while valid:
        val = input(feature + ': ')
        if not val.isdigit():valid = False
    predict[feature] = [float(val)]

# Make Predictions
predictions =classifier.predict(input_fn = lambda: input_fn(predict))
for pred_dict in predictions:
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

    print('predicition is "{}" ({:.1f}%)'.format(SPECIES[class_id], 100*probability))



