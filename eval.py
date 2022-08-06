import torch
import json
import numpy as np
import pandas as pd
from utils import calculate_mAP
import os
from pprint import PrettyPrinter

# Good formatting when printing the APs for each class and mAP
pp = PrettyPrinter()


def evaluate(loader, model, save_csv=False, verbose=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    detected_boxes = list()
    detected_labels = list()
    detected_scores = list()
    true_boxes = list()
    true_labels = list()
    true_difficulties = list()

    model.eval()
    with torch.no_grad():
        for i, (images, targets) in enumerate(loader):
            # Move to default device
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward prop
            res = model(images)

            for k in range(len(images)):
                det_boxes = res[k].get("boxes", None)
                det_labels = res[k].get("labels", None)
                det_scores = res[k].get("scores", None)
                # TODO What happens if one of them is None?

                detected_boxes.append(det_boxes)
                detected_labels.append(det_labels)
                detected_scores.append(det_scores)
                true_boxes.append(targets[k]['boxes'])
                true_labels.append(targets[k]['labels'])
                true_difficulties.append(torch.zeros_like(targets[k]['labels']))  # made up for using `calculate_mAP()`

            del images, res, det_boxes, det_labels, det_scores

        # Calculate mAP
        APs, mAP = calculate_mAP(detected_boxes, detected_labels, detected_scores, true_boxes, true_labels, true_difficulties)

    # Print AP for each class
    pp.pprint(APs)

    print('\nMean Average Precision (mAP): %.3f' % mAP)

    # TODO Continue from here -
    #   any other results? what current results mean?

    del detected_boxes, detected_labels, detected_scores, true_boxes, true_labels, true_difficulties
    torch.cuda.empty_cache()

# def evaluate(loader, model, save_csv=False, verbose=False):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     detected_boxes = list()
#     detected_labels = list()
#     detected_scores = list()
#     true_boxes = list()
#     true_labels = list()
#     true_difficulties = list()
#
#     model.eval()
#     with torch.no_grad():
#         for i, (images, targets) in enumerate(loader):
#             # Move to default device
#             images = [image.to(device) for image in images]
#             targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
#
#             # Forward prop
#             res = model(images)
#
#             for k in range(len(images)):
#                 det_boxes = res[k].get("boxes", None)
#                 det_labels = res[k].get("labels", None)
#                 det_scores = res[k].get("scores", None)
#                 # TODO What happens if one of them is None?
#
#                 detected_boxes.append(det_boxes)
#                 detected_labels.append(det_labels)
#                 detected_scores.append(det_scores)
#                 true_boxes.append(targets[k]['boxes'])
#                 true_labels.append(targets[k]['labels'])
#                 true_difficulties.append(torch.zeros_like(targets[k]['labels']))  # made up for using `calculate_mAP()`
#
#             del images, res, det_boxes, det_labels, det_scores
#
#     # TODO Continue going from here below, make sure calculate_mAP works
#
#     filenames = loader.dataset.filenames
#     imgs_orig_sizes = loader.dataset.sizes
#
#     # convert boxes back to their original sizes by the original width, height
#     predicted_boxes = [box * imgs_orig_sizes[i].to(device) / 224 for i, box in enumerate(detected_boxes)]
#
#     # convert to [x_min, y_min, w, h] format
#     predicted_boxes = [[box[0][0], box[0][1], box[0][2] - box[0][0], box[0][3] - box[0][1]] for box in predicted_boxes]
#
#     predicted_labels = ['True' if label == 1 else 'False' for label in detected_labels]
#
#     # take true boxes from the filenames with format [x_min, y_min, w, h]
#     true_boxes = [json.loads(filename.strip(".jpg").split("__")[1]) for filename in filenames]
#
#     true_labels = [filename.strip(".jpg").split("__")[2] for filename in filenames]
#     mean_accuracy = np.mean([pred == true for pred, true in zip(predicted_labels, true_labels)])
#
#     mean_iou = np.mean([calc_iou(true_box, torch.stack(pred_box).cpu().numpy())
#                         for true_box, pred_box in zip(true_boxes, predicted_boxes)])
#
#     if verbose:
#         print(f'IoU = {round(float(mean_iou), 4)}, Accuracy = {round(float(mean_accuracy), 4)}')
#
#     if save_csv:
#         results = pd.DataFrame(columns=['filename', 'x', 'y', 'w', 'h', 'proper_mask'])
#         results.filename = filenames
#         predicted_boxes = [torch.stack(pred_box).cpu().numpy() for pred_box in predicted_boxes]
#         results.x, results.y, results.w, results.h = zip(*predicted_boxes)
#         results.proper_mask = predicted_labels
#         results.to_csv(save_csv, index=False, header=True)
#         print(f'saved results to {os.path.join(os.getcwd(), str(save_csv))}')
#
#     del predicted_boxes, detected_boxes, detected_labels, detected_scores
#     torch.cuda.empty_cache()
#
#     return mean_accuracy, mean_iou
