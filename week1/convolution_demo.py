import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image in grayscale
i = cv2.imread("week1/stairs.png", cv2.IMREAD_GRAYSCALE)

# Check if image loaded successfully
if i is None:
    print("Error: Could not load image.")
    exit()

# Show original image
plt.figure(figsize=(6,6))
plt.imshow(i, cmap="gray")
plt.title("Original Image")
plt.axis("off")
plt.show()

# Copy image
i_transformed = np.copy(i)

# Image dimensions
size_x = i.shape[0]
size_y = i.shape[1]

# Convolution Filter (Horizontal Edge Detection)
filter = [
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
]

weight = 1

# Apply convolution
for x in range(1, size_x - 1):
    for y in range(1, size_y - 1):

        output_pixel = 0.0

        output_pixel += int(i[x-1, y-1]) * filter[0][0]
        output_pixel += int(i[x,   y-1]) * filter[0][1]
        output_pixel += int(i[x+1, y-1]) * filter[0][2]

        output_pixel += int(i[x-1, y]) * filter[1][0]
        output_pixel += int(i[x,   y]) * filter[1][1]
        output_pixel += int(i[x+1, y]) * filter[1][2]

        output_pixel += int(i[x-1, y+1]) * filter[2][0]
        output_pixel += int(i[x,   y+1]) * filter[2][1]
        output_pixel += int(i[x+1, y+1]) * filter[2][2]

        output_pixel *= weight

        # Clamp values between 0 and 255
        output_pixel = max(0, min(255, output_pixel))

        i_transformed[x, y] = output_pixel

# Display transformed image
plt.figure(figsize=(6,6))
plt.imshow(i_transformed, cmap="gray")
plt.title("After Convolution")
plt.axis("off")
plt.show()