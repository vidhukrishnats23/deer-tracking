import cv2
import numpy as np
import os

# Create a dummy image for testing
img = np.zeros((512, 512, 3), np.uint8)
cv2.line(img, (100, 100), (400, 400), (255, 255, 255), 5)

# Create a directory for test data if it doesn't exist
os.makedirs('tests/data', exist_ok=True)

# Save the image
cv2.imwrite('tests/data/test_line.png', img)
