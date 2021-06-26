import torch
import json
import numpy as np
import pandas as pd
from utils import calc_iou
import os


def evaluate(loader, model, save_csv=False, verbose=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_images_boxes = list()
    all_images_labels = list()
    all_images_scores = list()

    model.eval()
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            # Move to default device
            images = [image.to(device) for image in images]

            # Forward prop
            res = model(images)

            for k in range(len(images)):
                boxes = res[k].get("boxes", None)
                labels = res[k].get("labels", None)
                scores = res[k].get("scores", None)
                if boxes is not None and labels is not None and scores is not None \
                        and torch.numel(boxes) != 0 and torch.numel(labels) != 0 and torch.numel(scores) != 0:
                    # TODO think of a better solution to the most_left need
                    most_left_index = int(torch.sort(boxes, dim=0, descending=False)[1][0][0])
                    # [xmin, ymin, xmax, ymax] non-fractional
                    all_images_boxes.append(boxes[most_left_index].to(device))
                    all_images_labels.append(labels[most_left_index].to(device))
                    all_images_scores.append(scores[most_left_index].to(device))
                else:
                    all_images_boxes.append(torch.FloatTensor([0., 0., 300., 300.]).to(device))
                    all_images_labels.append(torch.IntTensor([0]).to(device))
                    all_images_scores.append(torch.FloatTensor([0.]).to(device))

    filenames = loader.dataset.images
    imgs_orig_sizes = loader.dataset.sizes

    # clamp to [min=0, max=300] all predicted boxes
    predicted_boxes = [box.clamp(0., 300.) for box in all_images_boxes]

    # convert boxes back to their original sizes by the original width, height
    predicted_boxes = [box * imgs_orig_sizes[i].to(device) / 300 for i, box in enumerate(all_images_boxes)]

    # convert to [x_min, y_min, w, h] format
    predicted_boxes = [[box[0][0], box[0][1], box[0][2] - box[0][0], box[0][3] - box[0][1]] for box in predicted_boxes]

    # TODO make sure what's the best guess
    predicted_labels = ['True' if label == 1 else 'False' for label in all_images_labels]

    # take true boxes from the filenames with format [x_min, y_min, w, h]
    true_boxes = [json.loads(filename.strip(".jpg").split("__")[1]) for filename in filenames]

    true_labels = [filename.strip(".jpg").split("__")[2] for filename in filenames]
    mean_accuracy = np.mean([pred == true for pred, true in zip(predicted_labels, true_labels)])

    mean_iou = np.mean([calc_iou(true_box, torch.stack(pred_box).cpu().numpy())
                        for true_box, pred_box in zip(true_boxes, predicted_boxes)])

    if verbose:
        print(f'IoU = {round(float(mean_iou), 4)}, Accuracy = {round(float(mean_accuracy), 4)}')

    if save_csv:
        results = pd.DataFrame(columns=['filename', 'x', 'y', 'w', 'h', 'proper_mask'])
        results.filename = filenames
        results.x, results.y, results.w, results.h = zip(*predicted_boxes)
        results.proper_mask = predicted_labels
        results.to_csv(save_csv, index=False, header=True)
        print(f'saved results to {os.path.join(os.getcwd(), str(save_csv))}')
    return mean_accuracy, mean_iou
