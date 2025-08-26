
# Drawing Alignment Tool

This tool aligns two versions of the same drawing using computer vision. It detects features using SIFT, matches them between images, and applies a constrained affine transformation to align the drawings.

## Usage
Using the comamnd line tool:

```bash
python align_drawings.py \
    --old_path drawings/test1_old.png \
    --new_path drawings/test1_new.png \
    --overlay_path drawings/test1_overlay.png \
    --show_overlay
```

Using the API:
```py
aligned_old_img = AlignDrawings(debug=args.debug)(old_img, new_img)
overlay = create_overlay_image(aligned_old_img, new_img)
```
