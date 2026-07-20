import os
import shutil
import random


# ==========================================
# DATASET LOCATION
# ==========================================

DATASET_PATH = r"C:\AIML\week1\dogs-vs-cats"

TRAIN_SOURCE = os.path.join(
    DATASET_PATH,
    "train",
    "train"
)


# ==========================================
# CHECK PATHS
# ==========================================

print("\n========== PATH CHECK ==========")

print("Dataset exists:",
      os.path.exists(DATASET_PATH))

print("Train folder exists:",
      os.path.exists(TRAIN_SOURCE))

print("================================")


if not os.path.exists(TRAIN_SOURCE):

    print("ERROR: Train folder not found")
    exit()



# ==========================================
# CREATE OUTPUT FOLDERS
# ==========================================

OUTPUT = r"C:\AIML\week1\cats-v-dogs"


folders = [

    "training",
    "testing",

    "training\\cats",
    "training\\dogs",

    "testing\\cats",
    "testing\\dogs"

]


for folder in folders:

    os.makedirs(
        os.path.join(
            OUTPUT,
            folder
        ),
        exist_ok=True
    )


# ==========================================
# FIND IMAGES
# ==========================================


cats = []
dogs = []


for filename in os.listdir(TRAIN_SOURCE):

    if filename.startswith("cat"):

        cats.append(filename)


    elif filename.startswith("dog"):

        dogs.append(filename)



print("\n========== DATA FOUND ==========")

print("Cats found:",
      len(cats))

print("Dogs found:",
      len(dogs))

print("================================")



# ==========================================
# COPY FUNCTION
# ==========================================


def split_data(files, train_folder, test_folder):

    random.shuffle(files)


    split = int(
        len(files) * 0.9
    )


    training_files = files[:split]

    testing_files = files[split:]



    for file in training_files:

        shutil.copy(

            os.path.join(
                TRAIN_SOURCE,
                file
            ),

            os.path.join(
                OUTPUT,
                train_folder,
                file
            )

        )



    for file in testing_files:

        shutil.copy(

            os.path.join(
                TRAIN_SOURCE,
                file
            ),

            os.path.join(
                OUTPUT,
                test_folder,
                file
            )

        )



# ==========================================
# SPLIT DATA
# ==========================================


split_data(
    cats,
    "training\\cats",
    "testing\\cats"
)


split_data(
    dogs,
    "training\\dogs",
    "testing\\dogs"
)



# ==========================================
# VERIFY RESULT
# ==========================================


print("\n========== FINAL DATA ==========")


print(
    "Training Cats:",
    len(
        os.listdir(
            OUTPUT + "\\training\\cats"
        )
    )
)


print(
    "Training Dogs:",
    len(
        os.listdir(
            OUTPUT + "\\training\\dogs"
        )
    )
)


print(
    "Testing Cats:",
    len(
        os.listdir(
            OUTPUT + "\\testing\\cats"
        )
    )
)


print(
    "Testing Dogs:",
    len(
        os.listdir(
            OUTPUT + "\\testing\\dogs"
        )
    )
)


print("\nDATA PREPARATION COMPLETE")