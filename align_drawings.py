import argparse
from dataclasses import dataclass

import cv2
import numpy as np

from estimate_affine import estimate_affine_partial_2d_constrained
from image_utils import create_overlay_image, image_to_grayscale, load_image
from visualization_utils import show_matched_features, show_overlay_image, show_side_by_side


class AlignDrawings:
    """
    AlignDrawings is a utility class to align two images using computer vision.

    Sample usage:
    old_img = cv2.imread(old_path)
    new_img = cv2.imread(new_path)
    aligned_old_image = AlignDrawings()(old_img, new_img)

    You can also pass a AlignDrawings.Config object with your configurations to customize the alignment algorithm.
    """

    @dataclass
    class Config:
        n_features: int = 10_000
        exclude_margin: float = 0.2
        ratio_threshold: float = 0.75
        ransac_reproj_threshold: float = 3.0
        max_iters: int = 2000
        confidence: float = 0.99
        scale_min: float = 0.5
        scale_max: float = 1.5
        rotation_deg_min: float = -15
        rotation_deg_max: float = 15

    def __init__(self, config=None, debug=False):
        self.config = config or self.Config()
        self.debug = debug

    def __call__(self, old_img: np.ndarray, new_img: np.ndarray) -> np.ndarray:
        old_gray = image_to_grayscale(old_img)
        new_gray = image_to_grayscale(new_img)

        if self.debug:
            show_side_by_side(old_gray, new_gray)

        print("Extracting SIFT features...")
        kp1, desc1 = self.extract_features_sift(old_gray)
        kp2, desc2 = self.extract_features_sift(new_gray)
        print(f"Found {len(kp1)} keypoints in old image and {len(kp2)} in new image")

        if desc1 is None or desc2 is None or len(kp1) < 2 or len(kp2) < 2:
            print("Not enough features detected in one of the images.")
            return

        print("Matching features with Ratio Test...")
        good_matches = self.match_features_ratio_test(desc1, desc2)
        print(f"Found {len(good_matches)} good matches after ratio test")

        # Debug visualization: show all good matches before homography
        if self.debug:
            print("\n=== DEBUG: Visualizing all good matches ===")
            show_matched_features(old_gray, new_gray, kp1, kp2, good_matches)

        print("Finding transformation...")
        matrix, mask = self.find_transformation(kp1, kp2, good_matches)

        if matrix is None:
            raise RuntimeError("Failed to find a valid transformation.")

        # Debug visualization: show inlier matches after homography
        if self.debug and mask is not None:
            print("\n=== DEBUG: Visualizing inlier matches after RANSAC ===")
            show_matched_features(old_gray, new_gray, kp1, kp2, good_matches, mask.flatten())

        output_shape = (new_img.shape[1], new_img.shape[0])
        transformed_img = self.apply_transformation(old_img, matrix, output_shape)

        return transformed_img

    def extract_features_sift(self, img_gray):
        detector = cv2.SIFT_create(nfeatures=self.config.n_features)

        # Create mask to exclude margin area if specified
        mask = None
        if self.config.exclude_margin:
            h, w = img_gray.shape
            margin = int(min(h, w) * self.config.exclude_margin)
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[margin : h - margin, margin : w - margin] = 255

        keypoints, descriptors = detector.detectAndCompute(img_gray, mask)
        return keypoints, descriptors

    def match_features_ratio_test(self, desc1, desc2, ratio_threshold=0.75):
        """Match SIFT features using Brute-Force and Lowe's Ratio Test"""
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        matches_knn = matcher.knnMatch(desc1, desc2, k=2)

        good_matches = []
        for m, n in matches_knn:
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)

        good_matches = sorted(good_matches, key=lambda x: x.distance)
        return good_matches

    def find_transformation(self, kp1, kp2, matches):
        """Find a transformation matrix for transformations without shear and perspective."""
        if len(matches) < 3:
            print(f"Not enough good matches found: {len(matches)}")
            return None, None

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        matrix, mask = estimate_affine_partial_2d_constrained(
            from_points=src_pts.reshape(-1, 2),
            to_points=dst_pts.reshape(-1, 2),
            ransac_reproj_threshold=self.config.ransac_reproj_threshold,
            max_iters=self.config.max_iters,
            confidence=self.config.confidence,
            scale_min=self.config.scale_min,
            scale_max=self.config.scale_max,
            rotation_deg_min=self.config.rotation_deg_min,
            rotation_deg_max=self.config.rotation_deg_max,
        )
        return matrix, mask

    def apply_transformation(self, img, matrix, output_shape):
        """Apply an affine transformation to the image."""
        if matrix is None:
            return img
        return cv2.warpAffine(img, matrix, output_shape)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Align drawings by comparing old and new versions")
    parser.add_argument("--old_path", help="Old image path.", required=True)
    parser.add_argument("--new_path", help="New image path.", required=True)
    parser.add_argument("--show_overlay", action="store_true", help="Show overlay image after alignment.")
    parser.add_argument("--aligned_path", help="Optional path to save the aligned old image.", default=None)
    parser.add_argument("--overlay_path", help="Optional path to save the overlayed image.", default=None)
    parser.add_argument("--debug", action="store_true", help="Enable debug mode to visualize matched features.")
    args = parser.parse_args()

    # Load images
    old_img = load_image(args.old_path)
    new_img = load_image(args.new_path)

    # Align images
    transformed_img = AlignDrawings(debug=args.debug)(old_img, new_img)
    if args.aligned_path:
        cv2.imwrite(args.aligned_path, transformed_img)
        print(f"Aligned image saved to: {args.aligned_path}")

    # Create image overlay
    overlay = create_overlay_image(transformed_img, new_img)
    if args.overlay_path:
        cv2.imwrite(args.overlay_path, overlay)
        print(f"Overlay image saved to: {args.overlay_path}")
    if args.show_overlay:
        show_overlay_image(overlay, "after alignment")


if __name__ == "__main__":
    main()
