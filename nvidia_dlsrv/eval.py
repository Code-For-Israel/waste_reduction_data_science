from tqdm import tqdm
import torch.utils.data
import numpy as np
import pandas as pd
import os
import json
import utils


def evaluate(loader, model, encoder, save_csv=False, verbose=False):
    """
    Evaluate.

    :param loader: DataLoader for test data, created with shuffle=False
    :param model: model
    :param encoder: encoder
    :param save_csv: False or path to save file with predicted results
    :param verbose: whether to print IoU and accuracy or not
    return mean_accuracy, mean_iou for all the data in `loader`
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Make sure it's in eval mode
    model.eval()

    # Lists to store detected boxes and labels
    det_boxes = list()
    det_labels = list()

    with torch.no_grad():
        # Batches
        for i, (images, boxes, labels) in enumerate(tqdm(loader, desc='Evaluating')):
            images = images.to(device)  # (N, 3, 300, 300)

            # Forward prop.
            predicted_locs, predicted_scores = model(images)

            # Get predictions
            ploc, plabel = predicted_locs.float(), predicted_scores.float()

            # Handle the batch of predictions produced
            # This is slow, but consistent with old implementation.
            for idx in range(ploc.shape[0]):
                # ease-of-use for specific predictions
                ploc_i = ploc[idx, :, :].unsqueeze(0)
                plabel_i = plabel[idx, :, :].unsqueeze(0)

                try:
                    result = encoder.decode_batch(ploc_i, plabel_i, criteria=0.50, max_output=200)[0]
                    # result is list of (bboxes_out, labels_out, scores_out)
                    det_boxes.append(result[0])
                    det_labels.append(result[1])
                except Exception as e:
                    print(f"\nException: {e}")
                    print("No object detected in idx: {}".format(idx))
                    # No object detected so append som box and label
                    # TODO YOTAM: This should be done in decode_batch!!!!!!!!! (return the same thing if no box found)
                    det_boxes.append(torch.tensor([0., 0., 1., 1.]))
                    det_labels.append(torch.tensor(2))

        filenames = loader.dataset.images
        imgs_orig_sizes = loader.dataset.sizes

        # convert from fractional to non-fractional [x_min, y_min, x_max, y_max]
        predicted_boxes = [box * imgs_orig_sizes[i].to(device) for i, box in enumerate(det_boxes)]

        # convert to [x_min, y_min, w, h] format
        predicted_boxes = [[float(box[0][0]),
                            float(box[0][1]),
                            float(box[0][2] - box[0][0]),
                            float(box[0][3] - box[0][1])] for box in predicted_boxes]

        # TODO make sure what's the best guess
        predicted_labels = ['True' if label == 1 else 'False' for label in det_labels]

        true_boxes = [json.loads(filename.strip(".jpg").split("__")[1]) for filename in filenames]
        true_labels = [filename.strip(".jpg").split("__")[2] for filename in filenames]
        mean_accuracy = np.mean([pred == true for pred, true in zip(predicted_labels, true_labels)])

        mean_iou = np.mean([utils.calc_iou(true_box, pred_box)
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
