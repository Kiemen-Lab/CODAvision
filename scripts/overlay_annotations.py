#!/usr/bin/env python3

import os
import pickle
import numpy as np
import cv2

# ---------------
#  USER SETTINGS
# ---------------
image_path = "/Users/tnewton3/Desktop/liver_tissue_data/10x/Liv-17_0001.tif"
pkl_path   = "/Users/tnewton3/Desktop/liver_tissue_data/data py/Liv-17_0001/annotations.pkl"
output_path= "/Users/tnewton3/Desktop/liver_tissue_data/10x/Liv-17_0001_annotated.tif"

# ---------------
#  LOAD IMAGE
# ---------------
# Read the original H&E image as BGR
image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
if image_bgr is None:
    raise FileNotFoundError(f"Could not open '{image_path}'")

height, width = image_bgr.shape[:2]
print(f"Loaded H&E image: {image_path}, shape: {height}x{width}")

# ---------------
#  LOAD ANNOTATIONS
# ---------------
with open(pkl_path, "rb") as f:
    data = pickle.load(f)

if "xyout" not in data:
    raise ValueError(f"No 'xyout' found in {pkl_path}; "
                     "annotation polygon coordinates not present.")

xyout = data["xyout"]  # shape (N, 4) => [layer_id, region_id, x, y]

# Optionally handle scaling if present
reduce_factor = data.get("reduce_annotations", 1.0)
# If you need to apply the scale:
# xyout[:, 2:4] = xyout[:, 2:4] / reduce_factor

# ---------------
#  CREATE A 1-CHANNEL "ANNOTATION MASK"
# ---------------
mask_annotated = np.zeros((height, width), dtype=np.uint8)  # 0 => no annotation

# ---------------
#  CHOOSE COLORS PER LAYER
# ---------------
layer_ids = np.unique(xyout[:, 0]).astype(int)
color_palette = [
    (  0,   0, 255),  # red
    (  0, 255,   0),  # green
    (255,   0,   0),  # blue
    (  0, 255, 255),  # yellow
    (255,   0, 255),  # magenta
    (255, 255,   0),  # cyan
]
layer_color_map = {}
for i, lid in enumerate(layer_ids):
    layer_color_map[lid] = color_palette[i % len(color_palette)]

# ---------------
#  DRAW POLYGONS
# ---------------
unique_pairs = np.unique(xyout[:, 0:2], axis=0)  # (layer_id, region_id)

for pair in unique_pairs:
    layer_id  = int(pair[0])
    region_id = pair[1]

    # Gather all points for (layer_id, region_id)
    mask_pts  = (xyout[:, 0] == layer_id) & (xyout[:, 1] == region_id)
    coords_xy = xyout[mask_pts, 2:4]
    pts = coords_xy.round().astype(int)
    if len(pts) < 3:
        continue

    # Convert to Nx1x2 for OpenCV
    pts_for_cv = pts.reshape((-1, 1, 2))

    # Fill mask
    cv2.fillPoly(mask_annotated, [pts_for_cv], 255)

    # Draw polygon outline in the original image
    color_bgr = layer_color_map.get(layer_id, (255, 255, 255))
    cv2.polylines(image_bgr, [pts_for_cv], isClosed=True, color=color_bgr, thickness=2)

# ---------------
#  DARKEN NON-ANNOTATED AREAS
# ---------------
alpha_dark = 0.4
dark_indices = (mask_annotated == 0)
image_bgr[dark_indices] = (image_bgr[dark_indices] * alpha_dark).astype(np.uint8)

# ---------------
#  ADD LEGEND
# ---------------
# We'll place it in the top-left corner. Adjust spacing as desired.
legend_x = 20      # left offset
legend_y = 20      # top offset
box_size = 20      # size of color box
margin   = 10      # vertical space between legend entries
font_scale = 0.6
font_thickness = 2

for i, lid in enumerate(layer_ids):
    # Rectangle for the color
    color = layer_color_map[lid]
    top_left = (legend_x, legend_y + i*(box_size + margin))
    bottom_right = (legend_x + box_size, legend_y + box_size + i*(box_size + margin))
    cv2.rectangle(image_bgr, top_left, bottom_right, color, -1)

    # Text label (e.g. "Layer X")
    text_x = legend_x + box_size + 10
    text_y = legend_y + box_size - 5 + i*(box_size + margin)
    cv2.putText(image_bgr,
                f"Layer {lid}",
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                font_thickness,
                cv2.LINE_AA)

# ---------------
#  SAVE THE RESULT WITH LZW COMPRESSION
# ---------------
# According to OpenCV docs, LZW is specified by the value `5` for TIFF compression.
# Make sure your OpenCV build supports TIFF and these flags.
cv2.imwrite(output_path, image_bgr, [cv2.IMWRITE_TIFF_COMPRESSION, 5])
print(f"Annotated image saved to: {output_path}")
