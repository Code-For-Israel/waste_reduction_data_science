from utils import *
from tqdm import tqdm
import torch.utils.data
import numpy as np
import pandas as pd


def evaluate(loader, model, min_score, save_csv=False, verbose=False):
    """
    Evaluate.

    :param loader: DataLoader for test data, created with shuffle=False
    :param model: model
    :param min_score: minimum score for detect_objects()
    :param save_csv: False or path to save file with predicted results
    :param verbose: whether to print IoU and accuracy or not
    return mean_accuracy, mean_iou for all the data in `loader`
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Make sure it's in eval mode
    model.eval()

    # Lists to store detected and true boxes, labels, scores
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()

    with torch.no_grad():
        # Batches
        for i, (images, boxes, labels) in enumerate(tqdm(loader, desc='Evaluating')):
            images = images.to(device)  # (N, 3, 300, 300)

            # Forward prop.
            predicted_locs, predicted_scores = model(images)

            # Detect objects in SSD output
            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs, predicted_scores,
                                                                                       min_score=min_score,
                                                                                       max_overlap=0.45,
                                                                                       top_k=200)
            # Evaluation MUST be at min_score=0.01, max_overlap=0.45, top_k=200 for
            # fair comparison with the paper's results and other repos
            # TODO YOTAM look what parameters we want, min_score=GAL 0.01 sounds really low

            # Store this batch's results for accuracy, IoU calculation
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]

            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            true_boxes.extend(boxes)
            true_labels.extend(labels)

        filenames = loader.dataset.images
        imgs_orig_sizes = loader.dataset.sizes

        # convert from fractional to non-fractional [x_min, y_min, x_max, y_max]
        predicted_boxes = [box * imgs_orig_sizes[i].to(device) for i, box in enumerate(det_boxes)]

        # convert to [x_min, y_min, w, h] format
        predicted_boxes = [[box[0][0], box[0][1], box[0][2] - box[0][0], box[0][3] - box[0][1]] for box in
                           predicted_boxes]

        # TODO make sure what's in `det_labels`. GAL `det_labels` contains 0 (background), 1 (proper), 2 (not proper).
        #  pay attention this will give label 'False' also if predicted background (0) - do we want this?
        predicted_labels = ['True' if label == 1 else 'False' for label in det_labels]

        # overwrite the true_boxes to take it from the filenames with format [x_min, y_min, w, h]
        true_boxes = [json.loads(filename.strip(".jpg").split("__")[1]) for filename in filenames]

        # TODO YOTAM verify this one,
        #  maybe it's better to compare `predicted_labels` and also get the true labels from filenames
        mean_accuracy = np.mean(torch.cat([(pred == true).cpu()
                                           for pred, true in zip(det_labels, true_labels)]).numpy())

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
