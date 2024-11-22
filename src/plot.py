import matplotlib.pyplot as plt
import numpy as np

def show_image_grid(images, n_row=4, n_col=4):
    """ Display a grid of images. """
    fig, axes = plt.subplots(n_row, n_col, figsize=(8, 8))
    axes = axes.flatten()  # Flatten the grid to make it easier to plot in a loop

    for i in range(n_row * n_col):
        image = images[i].numpy().transpose((1, 2, 0))  # Convert Tensor to numpy array
        axes[i].imshow(np.clip(image, 0, 1))  # Display image and clip values between 0 and 1
        axes[i].axis('off')  # Remove axis

    plt.tight_layout()
    plt.show()
