
# Drawing Alignment Tool

This tool aligns two versions of the same drawing using computer vision. It detects features using SIFT, matches them between images, and applies a constrained affine transformation to align the drawings.

## Usage

```bash
python align_drawings.py --old_path drawings/test1_old.png --new_path drawings/test1_new.png
```
