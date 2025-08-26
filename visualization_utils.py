import cv2
import matplotlib.pyplot as plt
import numpy as np


def show_side_by_side(old_img, new_img):
    """Display old and new images side by side for comparison"""
    plt.figure(figsize=(15, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(old_img, cv2.COLOR_BGR2RGB))
    plt.title("Old Image")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB))
    plt.title("New Image")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def show_matched_features(old_img, new_img, kp1, kp2, matches, mask=None):
    """Visualize matched features between two images"""
    # Filter matches based on mask if provided (inliers only)
    if mask is not None:
        filtered_matches = [matches[i] for i in range(len(matches)) if mask[i]]
        print(f"Visualizing {len(filtered_matches)} inlier matches out of {len(matches)} total matches")
    else:
        filtered_matches = matches
        print(f"Visualizing all {len(filtered_matches)} matches")

    # Draw matches
    img_matches = cv2.drawMatches(
        old_img,
        kp1,
        new_img,
        kp2,
        filtered_matches[:50],  # Limit to first 50 matches for clarity
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    # Display the result
    plt.figure(figsize=(20, 10))
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.title(f"Feature Matches (showing top 50 out of {len(filtered_matches)} matches)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def show_overlay_image(image, title):
    # Display the results
    plt.figure(figsize=(15, 12))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Overlay (Green=Old, Red=New) " + title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
