import cv2
import numpy as np
import os
from tqdm import tqdm

src_images_path = "../dataset/crop_images/"
destination_images_path = "../dataset/removed_background/"

files = os.listdir(src_images_path)
for i, filename in tqdm(enumerate(files)):
    # Load the image
    image = cv2.imread(f'{src_images_path}{filename}')

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define range for white background in HSV
    lower_white = np.array([0, 0, 0])   # Adjust these values as needed
    upper_white = np.array([180, 35, 255]) 

    # Create a mask to detect the white background
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Invert the mask to get the apple area
    apple_mask = cv2.bitwise_not(mask)

    # Apply the mask to the original image
    result = cv2.bitwise_and(image, image, mask=apple_mask)

    # Optional: Set the background to transparent (useful for saving as PNG)
    bgr_result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    bgr_result[:, :, 3] = apple_mask  # Set the alpha channel

    # Save or display the result
    cv2.imwrite(f'{destination_images_path}{filename}', bgr_result)
