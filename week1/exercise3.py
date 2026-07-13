import tensorflow as tf

print("TensorFlow Version:", tf.__version__)

# ----------------------------
# Load Fashion MNIST Dataset
# ----------------------------
mnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# ----------------------------
# Reshape Images for CNN
# ----------------------------
training_images = training_images.reshape(60000, 28, 28, 1)
test_images = test_images.reshape(10000, 28, 28, 1)

# ----------------------------
# Normalize Pixel Values
# ----------------------------
training_images = training_images / 255.0
test_images = test_images / 255.0

# ----------------------------
# CNN Model (3 Convolution Layers)
# ----------------------------
model = tf.keras.models.Sequential([

    # First Convolution Layer
    tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation='relu',
        input_shape=(28, 28, 1)
    ),

    # First Max Pooling
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    # Second Convolution Layer
    tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation='relu'
    ),

    # Second Max Pooling
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    # Third Convolution Layer (NEW)
    tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        activation='relu'
    ),

    # Third Max Pooling (NEW)
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    # Flatten
    tf.keras.layers.Flatten(),

    # Dense Hidden Layer
    tf.keras.layers.Dense(
        units=128,
        activation='relu'
    ),

    # Output Layer
    tf.keras.layers.Dense(
        units=10,
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

# ----------------------------
# Print Results
# ----------------------------
print("\n========== RESULTS ==========")
print(f"Test Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy * 100:.2f}%")
print("=============================")