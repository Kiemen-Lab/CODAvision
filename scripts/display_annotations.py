#!/usr/bin/env python3

import os
import pickle
import numpy as np
import cv2
import napari
import shapely.geometry
import shapely.ops

# ---------------
#  USER SETTINGS
# ---------------
image_path  = "/Users/tnewton3/Desktop/liver_tissue_data/10x/Liv-17_0001.tif"
pkl_path    = "/Users/tnewton3/Desktop/liver_tissue_data/data py/Liv-17_0001/annotations.pkl"

# ---------------
#  LOAD IMAGE
# ---------------
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
xyout[:, [2, 3]] = xyout[:, [3, 2]]  # swap columns <-> rows

# If there is a "reduce_annotations" factor, apply it:
reduce_factor = data.get("reduce_annotations", 1.0)
# If necessary, un-comment this to scale:
# xyout[:, 2:4] = xyout[:, 2:4] / reduce_factor

# ---------------
#  PREPARE COLORS FOR LAYERS
# ---------------
layer_ids = np.unique(xyout[:, 0]).astype(int)

# Some example colors (in BGR)
color_palette_bgr = [
    (  0,   0, 255),  # red
    (  0, 255,   0),  # green
    (255,   0,   0),  # blue
    (  0, 255, 255),  # yellow
    (255,   0, 255),  # magenta
    (255, 255,   0),  # cyan
]

layer_color_map = {}
for i, lid in enumerate(layer_ids):
    layer_color_map[lid] = color_palette_bgr[i % len(color_palette_bgr)]

# ---------------
#  BUILD POLYGON DATA FOR NAPARI
# ---------------
# We'll display each region as a polygon in a napari Shapes layer.
shapes_data = []  # list of Nx2 arrays, one per polygon
edge_colors = []
face_colors = []

unique_pairs = np.unique(xyout[:, 0:2], axis=0)  # (layer_id, region_id)

for pair in unique_pairs:
    layer_id = int(pair[0])
    region_id = pair[1]

    # Get the coordinates for (layer_id, region_id)
    mask_pts = (xyout[:, 0] == layer_id) & (xyout[:, 1] == region_id)
    coords_xy = xyout[mask_pts, 2:4]
    pts = coords_xy.round().astype(int)

    # We need at least 3 points to form a polygon
    if len(pts) < 3:
        continue

    # Convert to a shapely Polygon
    polygon = shapely.geometry.Polygon(pts)
    # Attempt to "clean" the polygon (fix self-intersections, etc.)
    # The buffer(0) trick often fixes minor self-intersections & duplicate points
    polygon_clean = polygon.buffer(0)

    # If the result is empty or not a valid polygon, skip
    if polygon_clean.is_empty or not polygon_clean.is_valid:
        print(f"Skipping invalid polygon for layer {layer_id}, region {region_id}")
        continue

    # For simple shapes, we can just use the outer boundary
    # (If there are holes, you'll need to handle them too, see below)
    exterior_coords = np.array(polygon_clean.exterior.coords)

    # If shapely introduced holes, you can also gather them by iterating
    # polygon_clean.interiors. But let's keep this example simple:

    # Convert color from BGR to normalized RGBA
    bgr = layer_color_map.get(layer_id, (255, 255, 255))
    color_rgb = (bgr[2] / 255.0, bgr[1] / 255.0, bgr[0] / 255.0)
    edge_color_rgba = color_rgb + (1.0,)
    face_color_rgba = color_rgb + (0.3,)

    shapes_data.append(exterior_coords)
    edge_colors.append(edge_color_rgba)
    face_colors.append(face_color_rgba)

# ---------------
#  LAUNCH NAPARI VIEWER
# ---------------
# Convert the original BGR image to RGB for display in napari
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

viewer = napari.Viewer()
viewer.add_image(image_rgb, name="H&E Image")

# Add polygons as a shapes layer
viewer.add_shapes(
    data=shapes_data,
    shape_type='polygon',
    edge_color=edge_colors,
    face_color=face_colors,
    edge_width=2,
    name='Annotations'
)

napari.run()
