import tensorflow as tf

# Load Fashion MNIST dataset
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# Normalize pixel values (0-255 -> 0-1)
training_images = training_images / 255.0
test_images = test_images / 255.0

# Build the Deep Neural Network
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(training_images, training_labels, epochs=5)

# Evaluate
test_loss, test_accuracy = model.evaluate(test_images, test_labels)

print(f"\nTest Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")