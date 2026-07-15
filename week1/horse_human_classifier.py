import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image


# =====================================================
# HORSE vs HUMAN IMAGE CLASSIFIER
# Week 1 - CNN Model
# =====================================================

print("TensorFlow Version:", tf.__version__)


# =====================================================
# STEP 1: Dataset Directories
# =====================================================

train_horse_dir = r"C:\AIML\week1\horse-or-human\horses"
train_human_dir = r"C:\AIML\week1\horse-or-human\humans"


# =====================================================
# STEP 2: Display Image File Names
# =====================================================

train_horse_names = os.listdir(train_horse_dir)
train_human_names = os.listdir(train_human_dir)

print("\n========== HORSE IMAGES ==========")
print(train_horse_names[:10])

print("\n========== HUMAN IMAGES ==========")
print(train_human_names[:10])


# =====================================================
# STEP 3: Count Images
# =====================================================

print("\n========== DATASET INFORMATION ==========")

print("Total Horse Images :", len(train_horse_names))
print("Total Human Images :", len(train_human_names))

print("=========================================\n")


# =====================================================
# STEP 4: Build CNN Model
# =====================================================

model = tf.keras.models.Sequential([

    # First Convolution Layer
    tf.keras.layers.Conv2D(
        filters=16,
        kernel_size=(3,3),
        activation='relu',
        input_shape=(300,300,3)
    ),

    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),



    # Second Convolution Layer
    tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=(3,3),
        activation='relu'
    ),

    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),



    # Third Convolution Layer
    tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(3,3),
        activation='relu'
    ),

    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),



    # Fourth Convolution Layer
    tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(3,3),
        activation='relu'
    ),

    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),



    # Fifth Convolution Layer
    tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(3,3),
        activation='relu'
    ),

    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),



    # Flatten Feature Maps
    tf.keras.layers.Flatten(),



    # Hidden Dense Layer
    tf.keras.layers.Dense(
        units=512,
        activation='relu'
    ),



    # Output Layer
    tf.keras.layers.Dense(
        units=1,
        activation='sigmoid'
    )

])


# =====================================================
# STEP 5: Display Model Summary
# =====================================================

print("\n========== CNN MODEL SUMMARY ==========\n")
model.summary()


# =====================================================
# STEP 6: Compile the Model
# =====================================================

model.compile(

    loss='binary_crossentropy',

    optimizer=RMSprop(
        learning_rate=0.001
    ),

    metrics=['accuracy']

)

print("\n========================================")
print("Model Compiled Successfully!")
print("========================================")


# =====================================================
# STEP 7: Create Image Generator
# =====================================================

train_datagen = ImageDataGenerator(

    rescale=1./255

)

train_generator = train_datagen.flow_from_directory(

    r"C:\AIML\week1\horse-or-human",

    target_size=(300,300),

    batch_size=128,

    class_mode='binary'

)

print("\n========================================")
print("Image Generator Created Successfully!")
print("========================================")


# =====================================================
# STEP 8: Train the Model
# =====================================================

history = model.fit(

    train_generator,

    steps_per_epoch=8,

    epochs=15,

    verbose=1

)

print("\n========================================")
print("Training Completed Successfully!")
print("========================================")


# =====================================================
# STEP 9: Test the Model
# =====================================================

# Change this path to any image you want to test
image_path = r"C:\AIML\week1\test_images\my_image.jpg"

if os.path.exists(image_path):

    # Load image
    img = image.load_img(
        image_path,
        target_size=(300,300)
    )

    # Convert to NumPy array
    x = image.img_to_array(img)

    # Normalize
    x = x / 255.0

    # Add batch dimension
    x = np.expand_dims(x, axis=0)

    # Predict
    prediction = model.predict(x)

    print("\nPrediction Value:", prediction[0][0])

    if prediction[0][0] > 0.5:
        print("Prediction: Human")
    else:
        print("Prediction: Horse")

else:

    print("\n========================================")
    print("Test image not found!")
    print("Please place an image at:")
    print(image_path)
    print("========================================")