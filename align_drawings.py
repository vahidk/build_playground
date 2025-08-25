import cv2
import matplotlib.pyplot as plt
import numpy as np


def load_images():
    """Load the old and new images"""
    old_path = "drawings/test1_old.png"
    new_path = "drawings/test1_new.png"

    old_img = cv2.imread(old_path)
    new_img = cv2.imread(new_path)

    if old_img is None or new_img is None:
        raise FileNotFoundError(f"Could not load images from {old_path} or {new_path}")

    # Convert to grayscale for feature detection
    old_gray = cv2.cvtColor(old_img, cv2.COLOR_BGR2GRAY)
    new_gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)

    return old_img, new_img, old_gray, new_gray


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
        mask[margin:h-margin, margin:w-margin] = 255
    
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

    if mask is not None:
        print(f"Homography found with {np.sum(mask)} inliers out of {len(matches)} matches")
    else:
        return None, None

    return matrix, mask


def apply_transformation(img, matrix, output_shape):
    """Apply perspective transformation to image"""
    if matrix is None:
        return img
    return cv2.warpPerspective(img, matrix, output_shape)


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
    print("Loading images...")
    old_img, new_img, old_gray, new_gray = load_images()

    # Margin parameter to exclude features near image borders (in pixels)
    h, w = old_gray.shape
    margin = int(min(h, w) * 0.2)

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

    print("Finding Homography transformation...")
    matrix, mask = find_transformation(kp1, kp2, good_matches)

    if matrix is not None:
        output_shape = (new_img.shape[1], new_img.shape[0])
        transformed_img = apply_transformation(old_img, matrix, output_shape)
        show_overlay_images(transformed_img, new_img, "after alignment")
    else:
        print("Failed to find a valid transformation.")


if __name__ == "__main__":
    main()
