"""
plot images with labels (rectangle, proper_mask) from a path
"""

import os
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
from matplotlib import patches
import constants

np.random.seed(42)


def parse_images_and_bboxes(image_dir, n):
    """
    Parse a directory with images.
    :param image_dir: Path to directory with images.
    :param n: number of images to plot
    :return: A list with (filename, image_id, bbox, proper_mask) for every image in the image_dir.
    """
    example_filenames = os.listdir(image_dir)
    data = []
    np.random.shuffle(example_filenames)
    for filename in example_filenames[:n]:
        image_id, bbox, proper_mask = filename.strip(".jpg").split("__")
        bbox = json.loads(bbox)
        proper_mask = True if proper_mask.lower() == "true" else False
        data.append((filename, image_id, bbox, proper_mask))
    return data


def show_images_and_bboxes(data, image_dir):
    """
    Plot images with bounding boxes. Predicts random bounding boxes and computes IoU.
    :param data: Iterable with (filename, image_id, bbox, proper_mask) structure.
    :param image_dir: Path to directory with images.
    :return: None
    """
    for filename, image_id, bbox, proper_mask in data:
        # Load image
        im = cv2.imread(os.path.join(image_dir, filename))
        # BGR to RGB
        im = im[:, :, ::-1]
        # Ground truth bbox
        x1, y1, w1, h1 = bbox
        # Predicted bbox
        # Plot image and bboxes
        fig, ax = plt.subplots()
        ax.imshow(im)
        rect = patches.Rectangle((x1, y1), w1, h1,
                                 linewidth=2, edgecolor='g', facecolor='none', label='ground-truth')
        ax.add_patch(rect)
        ax.add_patch(rect)
        fig.suptitle(f"proper_mask={proper_mask}")
        ax.axis('off')
        plt.show()


data = parse_images_and_bboxes(constants.TRAIN_IMG_PATH, n=50)
show_images_and_bboxes(data, constants.TRAIN_IMG_PATH)
