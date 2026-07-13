### change of convolutions from 32 to 64

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import models

# ----------------------------
# TensorFlow Version
# ----------------------------
print("TensorFlow Version:", tf.__version__)


# ----------------------------
# Load Fashion MNIST Dataset
# ----------------------------
mnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()


# ----------------------------
# Reshape Images for CNN
# CNN expects:
# (number of images, height, width, channels)
# ----------------------------
training_images = training_images.reshape(60000, 28, 28, 1)
test_images = test_images.reshape(10000, 28, 28, 1)


# ----------------------------
# Normalize Pixel Values
# Convert 0-255 into 0-1
# ----------------------------
training_images = training_images / 255.0
test_images = test_images / 255.0


# ----------------------------
# Build CNN Model
# ----------------------------
model = tf.keras.models.Sequential([

    # First Convolution Layer
    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation='relu',
        input_shape=(28, 28, 1)
    ),

    # Reduce image size
    tf.keras.layers.MaxPooling2D(2, 2),


    # Second Convolution Layer
    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation='relu'
    ),


    # Reduce image size again
    tf.keras.layers.MaxPooling2D(2, 2),


    # Convert feature maps into 1D vector
    tf.keras.layers.Flatten(),


    # Dense Hidden Layer
    tf.keras.layers.Dense(
        128,
        activation='relu'
    ),


    # Output Layer
    # 10 classes in Fashion MNIST
    tf.keras.layers.Dense(
        10,
        activation='softmax'
    )
])


# ----------------------------
# Compile Model
# ----------------------------
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# ----------------------------
# Model Summary
# ----------------------------
print("\n========== MODEL SUMMARY ==========\n")

model.summary()


# ----------------------------
# Train Model
# ----------------------------
print("\n========== TRAINING ==========\n")

model.fit(
    training_images,
    training_labels,
    epochs=5
)


# ----------------------------
# Evaluate Model
# ----------------------------
print("\n========== TESTING ==========\n")

test_loss, test_accuracy = model.evaluate(
    test_images,
    test_labels
)


print("\n========== RESULTS ==========")
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy * 100)
print("=============================")


# ==================================================
# VISUALIZE CONVOLUTIONS AND POOLING
# ==================================================


# Print first 100 test labels
print("\nFirst 100 Test Labels:")
print(test_labels[:100])


# Select three ankle boot images
FIRST_IMAGE = 0
SECOND_IMAGE = 23
THIRD_IMAGE = 28


# Select which filter to visualize
CONVOLUTION_NUMBER = 6


# Get outputs from every layer
layer_outputs = [
    layer.output for layer in model.layers
]


# Create a model that outputs intermediate layers
activation_model = models.Model(
    inputs=model.inputs,
    outputs=layer_outputs
)


# Create display area
f, axarr = plt.subplots(3, 4, figsize=(12, 8))


# Show feature maps
for x in range(4):

    # First image
    f1 = activation_model.predict(
        test_images[FIRST_IMAGE].reshape(1, 28, 28, 1),
        verbose=0
    )[x]

    axarr[0, x].imshow(
        f1[0, :, :, CONVOLUTION_NUMBER],
        cmap='inferno'
    )

    axarr[0, x].set_title(
        "Layer " + str(x)
    )

    axarr[0, x].grid(False)



    # Second image
    f2 = activation_model.predict(
        test_images[SECOND_IMAGE].reshape(1, 28, 28, 1),
        verbose=0
    )[x]

    axarr[1, x].imshow(
        f2[0, :, :, CONVOLUTION_NUMBER],
        cmap='inferno'
    )

    axarr[1, x].grid(False)



    # Third image
    f3 = activation_model.predict(
        test_images[THIRD_IMAGE].reshape(1, 28, 28, 1),
        verbose=0
    )[x]

    axarr[2, x].imshow(
        f3[0, :, :, CONVOLUTION_NUMBER],
        cmap='inferno'
    )

    axarr[2, x].grid(False)



plt.show()