import cv2
import numpy as np


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load images from {path}")
    return img


def image_to_grayscale(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    return gray


def create_overlay_image(old_img, new_img):
    """Visualize transformation results, drawing only inlier matches"""
    # Apply color tints and create overlay
    old_tinted = old_img.copy()
    new_tinted = new_img.copy()

    # Create masks for non-black pixels (assuming black is close to 0)
    old_mask = np.sum(old_img, axis=2) > 30
    new_mask = np.sum(new_img, axis=2) > 30

    # Apply green tint to black pixels in old image, keep white pixels white
    old_tinted = old_img.copy()
    old_tinted[~old_mask] = [0, 255, 0]

    # Apply red tint to black pixels in new image, keep white pixels white
    new_tinted = new_img.copy()
    new_tinted[~new_mask] = [0, 0, 255]

    # Create overlay by blending the two tinted images
    overlay = cv2.addWeighted(old_tinted, 0.5, new_tinted, 0.5, 0)
    return overlay
