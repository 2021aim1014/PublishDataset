import cv2
import os
from tqdm import tqdm

src_images_path = "../../dataset/images/"
destination_images_path = "../../dataset/crop_images/"

files = os.listdir(src_images_path)
for i, filename in tqdm(enumerate(files)):
    # Load the image
    image = cv2.imread(f'{src_images_path}{filename}')

    # Get the dimensions of the image
    height, width = image.shape[:2]

    # Calculate the pixels to crop (18% of the width and height)
    crop_pixels_h = int(0.18 * height)
    crop_pixels_w = int(0.18 * width)

    # Crop the image: remove 18% from each side
    cropped_image = image[crop_pixels_h:height - crop_pixels_h, crop_pixels_w:width - crop_pixels_w]

    # Save or display the cropped image
    cv2.imwrite(f'{destination_images_path}{filename}', cropped_image)

