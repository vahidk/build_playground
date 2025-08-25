import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np


def load_images(path: str):
    """Load the old and new images based on the provided image name

    Args:
        path: Path to the image file
    """
    img = cv2.imread(path)

    if img is None:
        raise FileNotFoundError(f"Could not load images from {path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)

    return img, gray


def extract_features_sift(img_gray, margin=0):
    """Extract SIFT features for higher accuracy

    Args:
        img_gray: Grayscale image for feature detection
        margin: Margin size in pixels to exclude from feature detection around image borders
    """
    detector = cv2.SIFT_create(nfeatures=2000)

    # Create mask to exclude margin area if specified
    mask = None
    if margin > 0:
        h, w = img_gray.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[margin : h - margin, margin : w - margin] = 255

    keypoints, descriptors = detector.detectAndCompute(img_gray, mask)
    return keypoints, descriptors


def match_features_ratio_test(desc1, desc2):
    """Match SIFT features using Brute-Force and Lowe's Ratio Test"""
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches_knn = matcher.knnMatch(desc1, desc2, k=2)

    good_matches = []
    ratio_thresh = 0.75
    for m, n in matches_knn:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    good_matches = sorted(good_matches, key=lambda x: x.distance)
    return good_matches


def find_transformation(kp1, kp2, matches):
    """Find Homography transformation matrix for best accuracy"""
    if len(matches) < 4:
        print(f"Not enough good matches found: {len(matches)}")
        return None, None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=3.0)
    return matrix, mask


def apply_transformation(img, matrix, output_shape):
    """Apply perspective transformation to image"""
    if matrix is None:
        return img
    return cv2.warpPerspective(img, matrix, output_shape)


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


def visualize_matched_features(old_img, new_img, kp1, kp2, matches, mask=None):
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


def show_overlay_images(old_img, new_img, title):
    """Visualize transformation results, drawing only inlier matches"""
    # Apply color tints and create overlay
    old_tinted = old_img.copy()
    new_tinted = new_img.copy()

    # Create masks for non-black pixels (assuming black is close to 0)
    old_mask = np.sum(old_img, axis=2) > 30  # Pixels that are not black
    new_mask = np.sum(new_img, axis=2) > 30  # Pixels that are not black

    # Apply green tint to black pixels in old image, keep white pixels white
    old_tinted = old_img.copy()
    old_tinted[~old_mask] = [0, 255, 0]  # Make black pixels green (BGR format)

    # Apply red tint to black pixels in new image, keep white pixels white
    new_tinted = new_img.copy()
    new_tinted[~new_mask] = [0, 0, 255]  # Make black pixels red (BGR format)

    # Create overlay by blending the two tinted images
    overlay = cv2.addWeighted(old_tinted, 0.5, new_tinted, 0.5, 0)

    # Display the results
    plt.figure(figsize=(15, 12))
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title("Overlay (Green=Old, Red=New) " + title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Align drawings by comparing old and new versions")
    parser.add_argument("--old_path", help="Old image path.")
    parser.add_argument("--new_path", help="New image path.")
    parser.add_argument("--margin", type=float, default=0.2, help="Margin as fraction of image size.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode to visualize matched features.")
    args = parser.parse_args()

    # Load images
    old_img, old_gray = load_images(args.old_path)
    new_img, new_gray = load_images(args.new_path)

    if args.debug:
        show_side_by_side(old_gray, new_gray)

    # Margin parameter to exclude features near image borders (in pixels)
    h, w = old_gray.shape
    margin = int(min(h, w) * args.margin)

    print(f"\n=== Using SIFT + Ratio Test + Homography with {margin}px margin ===")

    print("Extracting SIFT features...")
    kp1, desc1 = extract_features_sift(old_gray, margin)
    kp2, desc2 = extract_features_sift(new_gray, margin)
    print(f"Found {len(kp1)} keypoints in old image and {len(kp2)} in new image")

    if desc1 is None or desc2 is None or len(kp1) < 2 or len(kp2) < 2:
        print("Not enough features detected in one of the images.")
        return

    print("Matching features with Ratio Test...")
    good_matches = match_features_ratio_test(desc1, desc2)
    print(f"Found {len(good_matches)} good matches after ratio test")

    # Debug visualization: show all good matches before homography
    if args.debug:
        print("\n=== DEBUG: Visualizing all good matches ===")
        visualize_matched_features(old_img, new_img, kp1, kp2, good_matches)

    print("Finding Homography transformation...")
    matrix, mask = find_transformation(kp1, kp2, good_matches)

    if matrix is None:
        print("Failed to find a valid transformation.")
        return

    # Debug visualization: show inlier matches after homography
    if args.debug and mask is not None:
        print("\n=== DEBUG: Visualizing inlier matches after RANSAC ===")
        visualize_matched_features(old_img, new_img, kp1, kp2, good_matches, mask)

    output_shape = (new_img.shape[1], new_img.shape[0])
    transformed_img = apply_transformation(old_img, matrix, output_shape)
    show_overlay_images(transformed_img, new_img, "after alignment")


if __name__ == "__main__":
    main()
