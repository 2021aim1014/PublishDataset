import cv2
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

complete_dataset_path = "../../dataset/complete_dataset/"
cropped_image_path = "../../dataset/cropped_images/"
files = os.listdir(complete_dataset_path)
for i, filename in tqdm(enumerate(files)):
    image = cv2.imread(f"{complete_dataset_path}/{filename}")

    # Get the dimensions of the image
    height, width = image.shape[:2]

    # Calculate the pixels to crop (18% of the width and height)
    crop_pixels_h = int(0.26 * height)
    crop_pixels_w = int(0.18 * width)
    cropped_image = image[crop_pixels_h:height - crop_pixels_h, crop_pixels_w:width - crop_pixels_w]

    # Edge detection: gray scale -> blur -> edges
    gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 1.0)
    edges = cv2.Canny(blurred_image, 10, 0)

    # # Crop the images based on boundary from edges
    _edges = np.sum(edges>0, axis=1)
    _edges_shift = np.roll(_edges, 2)
    total = _edges + _edges_shift
    row_start, row_end = np.where(total > 50)[0][[0, -1]]

    _edges = np.sum(edges>0, axis=0)
    _edges_shift = np.roll(_edges, 2)
    total = _edges + _edges_shift
    col_start, col_end = np.where(total > 50)[0][[0, -1]]
    
    cropped_image = cropped_image[row_start-1:row_end + 1, col_start-1:col_end + 1]
    cv2.imwrite(f"{cropped_image_path}/{filename}", cropped_image)
