import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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


    # First Convolution

    tf.keras.layers.Conv2D(

        16,

        (3,3),

        activation="relu",

        input_shape=(150,150,3)

    ),


    tf.keras.layers.MaxPooling2D(

        2,2

    ),



    # Second Convolution

    tf.keras.layers.Conv2D(

        32,

        (3,3),

        activation="relu"

    ),


    tf.keras.layers.MaxPooling2D(

        2,2

    ),



    # Third Convolution

    tf.keras.layers.Conv2D(

        64,

        (3,3),

        activation="relu"

    ),


    tf.keras.layers.MaxPooling2D(

        2,2

    ),



    # Flatten

    tf.keras.layers.Flatten(),



    # Dense Layer

    tf.keras.layers.Dense(

        512,

        activation="relu"

    ),



    # Output Layer

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



print("\nModel compiled successfully!")






# =====================================================
# STEP 5: IMAGE DATA GENERATORS
# =====================================================


train_datagen = ImageDataGenerator(

    rescale=1.0/255

)



validation_datagen = ImageDataGenerator(

    rescale=1.0/255

)




# Training data

train_generator = train_datagen.flow_from_directory(

    TRAINING_DIR,

    target_size=(150,150),

    batch_size=32,

    class_mode="binary"

)




# Validation data

validation_generator = validation_datagen.flow_from_directory(

    TESTING_DIR,

    target_size=(150,150),

    batch_size=32,

    class_mode="binary"

)



print("\n========== CLASS LABELS ==========")

print(train_generator.class_indices)

print("==================================")






# =====================================================
# STEP 6: TRAIN THE MODEL
# =====================================================



history = model.fit(

    train_generator,

    epochs=5,

    validation_data=validation_generator,

    verbose=1

)



print("\nTraining completed successfully!")







# =====================================================
# STEP 7: VISUALIZE RESULTS
# =====================================================



accuracy = history.history["accuracy"]

validation_accuracy = history.history["val_accuracy"]


loss = history.history["loss"]

validation_loss = history.history["val_loss"]



epochs = range(1, len(accuracy)+1)





# -----------------------------
# Accuracy Graph
# -----------------------------


plt.figure(figsize=(8,5))


plt.plot(

    epochs,

    accuracy,

    label="Training Accuracy"

)


plt.plot(

    epochs,

    validation_accuracy,

    label="Validation Accuracy"

)


plt.title(

    "Training and Validation Accuracy"

)


plt.xlabel(

    "Epoch"

)


plt.ylabel(

    "Accuracy"

)


plt.legend()


plt.grid(True)


plt.show()






# -----------------------------
# Loss Graph
# -----------------------------


plt.figure(figsize=(8,5))


plt.plot(

    epochs,

    loss,

    label="Training Loss"

)


plt.plot(

    epochs,

    validation_loss,

    label="Validation Loss"

)



plt.title(

    "Training and Validation Loss"

)



plt.xlabel(

    "Epoch"

)



plt.ylabel(

    "Loss"

)



plt.legend()



plt.grid(True)



plt.show()




print("\nGraphs generated successfully!")


# =====================================================
# STEP 8: TEST YOUR MODEL
# =====================================================


from tensorflow.keras.preprocessing import image


print("\n========== MODEL TESTING ==========")



# Put your image path here

test_image_path = r"C:\AIML\week1\test_images\my_image.jpg"



# Check if image exists

if os.path.exists(test_image_path):


    # Load image

    img = image.load_img(

        test_image_path,

        target_size=(150,150)

    )



    # Convert image to array

    img_array = image.img_to_array(img)



    # Normalize pixels

    img_array = img_array / 255.0



    # Add batch dimension

    img_array = np.expand_dims(

        img_array,

        axis=0

    )



    # Make prediction

    prediction = model.predict(

        img_array

    )



    print("\nPrediction value:")

    print(prediction[0][0])



    # Classification

    if prediction[0][0] > 0.5:

        print("Prediction: DOG")

    else:

        print("Prediction: CAT")



else:


    print("Image not found!")

    print("Place your image here:")

    print(test_image_path)



print("\n===================================")
print("Testing Completed!")
print("===================================")