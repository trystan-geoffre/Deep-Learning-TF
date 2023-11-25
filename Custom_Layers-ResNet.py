# Import Libraries
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

# Load and Preprocess Dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32')/255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32')/255.0

# Custom CNN and Residual Block Layers
class CNNBlock(layers.Layer):
    def __init__(self, out_channels, kernel_size = 3):
        super(CNNBlock, self).__init__()
        self.conv = layers.Conv2D(out_channels, kernel_size, padding='same')
        self.bn = layers.BatchNormalization()

    def call(self, input_tensor, training = False):
        x = self.conv(input_tensor)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        return x

class ResBlock(layers.Layer):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.cnn1 = CNNBlock(channels[0])
        self.cnn2 = CNNBlock(channels[1])
        self.cnn3 = CNNBlock(channels[2])
        self.pooling = layers.MaxPooling2D()
        self.identity_mapping = layers.Conv2D(channels[1], 1, padding='same')

    def call(self, input_tensor, training = False):
        x = self.cnn1(input_tensor, training=training)
        x = self.cnn2(x, training = training)
        x = self.cnn3(
            x + self.identity_mapping(input_tensor), training=training
        )
        return self.pooling(x)

# ResNet-Like Model
class ResNet_Like(keras.Model):
    def __init__(self, num_classes=10):
        super(ResNet_Like, self).__init__()
        self.block1 = ResBlock([32, 32, 64])
        self.block2 = ResBlock([128, 128, 256])
        self.block3 = ResBlock([128, 256, 512])
        self.pool = layers.GlobalAveragePooling2D()
        self.classifier = layers.Dense(num_classes)

    def call(self, input_tensor, training = False):
        x = self.block1(input_tensor, training=training)
        x = self.block2(x, training = training)
        x = self.block3(x, training = training)
        x = self.pool(x)
        return self.classifier(x)

    def model(self):
        x = keras.Input(shape=(28, 28, 1))
        return keras.Model(inputs=[x], outputs = self.call(x))

# Model Compilation
model = ResNet_Like(num_classes=10)
model.compile(
    optimizer=keras.optimizers.legacy.Adam(),
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics = ['accuracy'],
)

# Training
model.fit(x_train, y_train, batch_size=64, epochs=1, verbose=1)
print(model.model().summary())

# Model Evaluation
model.evaluate(x_test, y_test, batch_size=64, verbose=1)

