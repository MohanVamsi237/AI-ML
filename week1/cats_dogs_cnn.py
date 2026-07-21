import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

import matplotlib.pyplot as plt


# =====================================================
# CATS VS DOGS CNN CLASSIFIER
# Week 1 AIML Project
# =====================================================


print("TensorFlow Version:", tf.__version__)



# =====================================================
# STEP 1: DATASET PATHS
# =====================================================


TRAINING_DIR = r"C:\AIML\week1\cats-v-dogs\training"

TESTING_DIR = r"C:\AIML\week1\cats-v-dogs\testing"



print("\n========== PATH CHECK ==========")

print(
    "Training folder exists:",
    os.path.exists(TRAINING_DIR)
)

print(
    "Testing folder exists:",
    os.path.exists(TESTING_DIR)
)

print("================================")



if not os.path.exists(TRAINING_DIR):

    print("Training folder not found!")

    exit()



if not os.path.exists(TESTING_DIR):

    print("Testing folder not found!")

    exit()



# =====================================================
# STEP 2: DEFINE CNN MODEL
# =====================================================


model = tf.keras.models.Sequential([



    # Convolution Layer 1

    tf.keras.layers.Conv2D(

        16,

        (3,3),

        activation="relu",

        input_shape=(150,150,3)

    ),



    tf.keras.layers.MaxPooling2D(

        2,2

    ),



    # Convolution Layer 2

    tf.keras.layers.Conv2D(

        32,

        (3,3),

        activation="relu"

    ),



    tf.keras.layers.MaxPooling2D(

        2,2

    ),



    # Convolution Layer 3

    tf.keras.layers.Conv2D(

        64,

        (3,3),

        activation="relu"

    ),



    tf.keras.layers.MaxPooling2D(

        2,2

    ),



    # Convert feature maps into vector

    tf.keras.layers.Flatten(),



    # Dense layer

    tf.keras.layers.Dense(

        512,

        activation="relu"

    ),



    # Output

    # 0 = Cat
    # 1 = Dog

    tf.keras.layers.Dense(

        1,

        activation="sigmoid"

    )

])



# =====================================================
# STEP 3: MODEL SUMMARY
# =====================================================


print("\n========== MODEL SUMMARY ==========")

model.summary()



# =====================================================
# STEP 4: COMPILE MODEL
# =====================================================


model.compile(

    optimizer=RMSprop(

        learning_rate=0.001

    ),

    loss="binary_crossentropy",

    metrics=["accuracy"]

)



print("\nModel compiled successfully")



# =====================================================
# STEP 5: CREATE IMAGE GENERATORS
# =====================================================


train_datagen = ImageDataGenerator(

    rescale=1.0/255

)



validation_datagen = ImageDataGenerator(

    rescale=1.0/255

)



# Training images

train_generator = train_datagen.flow_from_directory(

    TRAINING_DIR,

    batch_size=32,

    class_mode="binary",

    target_size=(150,150)

)



# Testing images

validation_generator = validation_datagen.flow_from_directory(

    TESTING_DIR,

    batch_size=32,

    class_mode="binary",

    target_size=(150,150)

)



print("\n========== CLASS LABELS ==========")

print(train_generator.class_indices)

print("===================================")



# =====================================================
# STEP 6: TRAIN THE MODEL
# =====================================================


history = model.fit(

    train_generator,

    epochs=5,

    verbose=1,

    validation_data=validation_generator

)



print("\nTraining Completed Successfully")



# =====================================================
# STEP 7: PLOT ACCURACY AND LOSS
# =====================================================


acc = history.history["accuracy"]

val_acc = history.history["val_accuracy"]


loss = history.history["loss"]

val_loss = history.history["val_loss"]



epochs = range(len(acc))



# Accuracy graph

plt.figure(figsize=(8,5))


plt.plot(

    epochs,

    acc,

    label="Training Accuracy"

)


plt.plot(

    epochs,

    val_acc,

    label="Validation Accuracy"

)


plt.title(

    "Training and Validation Accuracy"

)


plt.legend()

plt.show()



# Loss graph

plt.figure(figsize=(8,5))


plt.plot(

    epochs,

    loss,

    label="Training Loss"

)


plt.plot(

    epochs,

    val_loss,

    label="Validation Loss"

)


plt.title(

    "Training and Validation Loss"

)


plt.legend()

plt.show()



# =====================================================
# STEP 8: TEST YOUR OWN IMAGE
# =====================================================


test_image = r"C:\AIML\week1\test_images\my_image.jpg"



if os.path.exists(test_image):


    img = image.load_img(

        test_image,

        target_size=(150,150)

    )


    img_array = image.img_to_array(img)


    img_array = img_array / 255.0



    img_array = np.expand_dims(

        img_array,

        axis=0

    )



    prediction = model.predict(img_array)



    print("\nPrediction value:")

    print(prediction[0][0])



    if prediction[0][0] > 0.5:

        print("Prediction: DOG")


    else:

        print("Prediction: CAT")



else:

    print("\nNo test image found")

    print(

        "Place an image here:",

        test_image

    )



# =====================================================
# STEP 9: SAVE MODEL
# =====================================================


model.save(

    "cats_dogs_classifier.keras"

)



print("\nModel saved successfully")

print("\n========== PROGRAM COMPLETE ==========")