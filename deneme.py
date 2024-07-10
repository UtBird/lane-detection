# prompt: slide window method ve histogram grafiği kullanark gelen yol fotoğrafındaki şeritleri algılayıp birini yeşil diğerinide kırmızı olarka çizebilcek koud yazar mısısn 

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('images.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply a Gaussian blur to the image
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Threshold the image to create a binary image
thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)[1]

# Find the contours in the image
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a histogram of the contour lengths
hist = np.zeros(image.shape[1])
for contour in contours:
  hist[contour[:, 0, 0]] += 1

# Find the two largest peaks in the histogram
peaks = np.argsort(hist)[::-1][:2]

# Create a mask for each of the two largest peaks
mask1 = np.zeros_like(thresh)
mask1[hist == peaks[0]] = 254
mask2 = np.zeros_like(thresh)
mask2[hist == peaks[1]] = 255

# Apply the masks to the image
masked_image1 = cv2.bitwise_and(image, image, mask=mask1)
masked_image2 = cv2.bitwise_and(image, image, mask=mask2)

# Draw the two largest contours in green and red
cv2.drawContours(masked_image1, contours, peaks[0], (0, 255, 0), 2)
cv2.drawContours(masked_image2, contours, peaks[1], (0, 0, 255), 2)

# Show the results
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title('Original Image')
plt.subplot(1, 3, 2)
plt.imshow(masked_image1)
plt.title('Green Lane')
plt.subplot(1, 3, 3)
plt.imshow(masked_image2)
plt.title('Red Lane')
plt.show()
