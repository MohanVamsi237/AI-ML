import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load grayscale image
image = cv2.imread("week1/stairs.png", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Image not found!")
    exit()

# Convert to int32 (prevents overflow during convolution)
image = image.astype(np.int32)

# Copy image
i_transformed = np.copy(image)

size_x = image.shape[0]
size_y = image.shape[1]

# Horizontal Edge Detection Filter (Sobel Y)
filter = [
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
]

weight = 1

# ------------------------
# Apply Convolution
# ------------------------

for x in range(1, size_x - 1):
    for y in range(1, size_y - 1):

        output_pixel = 0.0

        output_pixel += image[x-1, y-1] * filter[0][0]
        output_pixel += image[x,   y-1] * filter[0][1]
        output_pixel += image[x+1, y-1] * filter[0][2]

        output_pixel += image[x-1, y] * filter[1][0]
        output_pixel += image[x,   y] * filter[1][1]
        output_pixel += image[x+1, y] * filter[1][2]

        output_pixel += image[x-1, y+1] * filter[2][0]
        output_pixel += image[x,   y+1] * filter[2][1]
        output_pixel += image[x+1, y+1] * filter[2][2]

        output_pixel *= weight

        output_pixel = max(0, min(255, output_pixel))

        i_transformed[x, y] = output_pixel

# ------------------------
# Show Convolution Output
# ------------------------

plt.figure(figsize=(6,6))
plt.imshow(i_transformed, cmap="gray")
plt.title("After Convolution")
plt.axis("off")
plt.show()

# ------------------------
# Max Pooling
# ------------------------

new_x = size_x // 2
new_y = size_y // 2

newImage = np.zeros((new_x, new_y))

for x in range(0, size_x - 1, 2):
    for y in range(0, size_y - 1, 2):

        pixels = [
            i_transformed[x, y],
            i_transformed[x+1, y],
            i_transformed[x, y+1],
            i_transformed[x+1, y+1]
        ]

        newImage[x//2, y//2] = max(pixels)

# ------------------------
# Show Pooling Result
# ------------------------

plt.figure(figsize=(6,6))
plt.imshow(newImage, cmap="gray")
plt.title("After Max Pooling")
plt.axis("off")
plt.show()