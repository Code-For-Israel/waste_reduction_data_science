# from utils import *
from tqdm import tqdm
import torch.utils.data
import numpy as np
import pandas as pd
import os
import json
import nvidia.utils as utils

def evaluate_nvidia(loader, model, encoder, min_score, topk, save_csv=False, verbose=False):
    """
    Evaluate.

    :param loader: DataLoader for test data, created with shuffle=False
    :param model: model
    :param min_score: minimum score for detect_objects()
    :param topk: take k top boxes by scores per image
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
            # det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs, predicted_scores,
            #                                                                            min_score=min_score,
            #                                                                            max_overlap=0.45,
            #                                                                            top_k=topk)

            # Get predictions # TODO from here new
            ploc, plabel = predicted_locs.float(), predicted_scores.float()

            # Handle the batch of predictions produced
            # This is slow, but consistent with old implementation.
            for idx in range(ploc.shape[0]):
                # ease-of-use for specific predictions
                ploc_i = ploc[idx, :, :].unsqueeze(0)
                plabel_i = plabel[idx, :, :].unsqueeze(0)

                ploc_i = ploc_i.permute(0, 2, 1)  # [1, 8732, 4] -> [1, 4, 8732]
                plabel_i = plabel_i.permute(0, 2, 1)  # [1, 8732, 3] -> [1, 3, 8732]
                try:
                    result = encoder.decode_batch(ploc_i, plabel_i, criteria=0.50, max_output=200)[0]
                    # result is list of (bboxes_out, labels_out, scores_out)s
                    det_boxes.append(result[0])
                    det_labels.append(result[1])
                    det_scores.append(result[2])
                except:
                    print("")
                    print("No object detected in idx: {}".format(idx))
                    continue

            # Evaluation MUST be at min_score=0.01, max_overlap=0.45, top_k=200 for
            # fair comparison with the paper's results and other repos
            # TODO YOTAM look what parameters we want, min_score=GAL 0.01 sounds really low



            # Store this batch's results for accuracy, IoU calculation
            boxes = [b.to(device) for b in boxes]  # list of torch.Size([1, 4])
            labels = [l.to(device) for l in labels]  # list of torch.Size([1])

            true_boxes.append(boxes)
            true_labels.append(labels)

        # htot, wtot = img_size[0][idx].item(), img_size[1][idx].item()
        # loc, label, prob = [r.cpu().numpy() for r in result]
        # for loc_, label_, prob_ in zip(loc, label, prob):
        #     ret.append([img_id[idx], loc_[0] * wtot, \
        #                 loc_[1] * htot,
        #                 (loc_[2] - loc_[0]) * wtot,
        #                 (loc_[3] - loc_[1]) * htot,
        #                 prob_,
        #                 inv_map[label_]])
        #
        # # Now we have all predictions from this rank, gather them all together
        # # if necessary
        # ret = np.array(ret).astype(np.float32)

        filenames = loader.dataset.images
        imgs_orig_sizes = loader.dataset.sizes

        # convert from fractional to non-fractional [x_min, y_min, x_max, y_max]
        predicted_boxes = [box * imgs_orig_sizes[i].to(device) for i, box in enumerate(det_boxes)]

        # convert to [x_min, y_min, w, h] format
        predicted_boxes = [[box[0][0], box[0][1], box[0][2] - box[0][0], box[0][3] - box[0][1]] for box in
                           predicted_boxes]

        # TODO make sure what's the best guess
        predicted_labels = ['True' if label == 1 else 'False' for label in det_labels]

        # overwrite the true_boxes to take it from the filenames with format [x_min, y_min, w, h]
        true_boxes = [json.loads(filename.strip(".jpg").split("__")[1]) for filename in filenames]

        true_labels = ['True' if label == 1 else 'False' for label in torch.cat(true_labels[0])]
        mean_accuracy = np.mean([pred == true for pred, true in zip(predicted_labels, true_labels)])

        mean_iou = np.mean([utils.calc_iou(true_box, torch.stack(pred_box).cpu().numpy())
                            for true_box, pred_box in zip(true_boxes, predicted_boxes)])

        predicted_boxes = [torch.stack(pred_box).cpu().numpy() for pred_box in predicted_boxes]
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


#############


# def evaluate(model, coco, cocoGt, encoder, inv_map, args):
#     ret = []
#
#     # for idx, image_id in enumerate(coco.img_keys):
#     for nbatch, (img, img_id, img_size, _, _) in enumerate(coco):
#         print("Parsing batch: {}/{}".format(nbatch, len(coco)), end='\r')
#         with torch.no_grad():
#             inp = img.cuda()
#             if args.amp:
#                 inp = inp.half()
#
#             # Get predictions
#             ploc, plabel = model(inp)
#             ploc, plabel = ploc.float(), plabel.float()
#
#             # Handle the batch of predictions produced
#             # This is slow, but consistent with old implementation.
#             for idx in range(ploc.shape[0]):
#                 # ease-of-use for specific predictions
#                 ploc_i = ploc[idx, :, :].unsqueeze(0)
#                 plabel_i = plabel[idx, :, :].unsqueeze(0)
#
#                 try:
#                     result = encoder.decode_batch(ploc_i, plabel_i, 0.50, 200)[0]
#                 except:
#                     # raise
#                     print("")
#                     print("No object detected in idx: {}".format(idx))
#                     continue
#
#                 htot, wtot = img_size[0][idx].item(), img_size[1][idx].item()
#                 loc, label, prob = [r.cpu().numpy() for r in result]
#                 for loc_, label_, prob_ in zip(loc, label, prob):
#                     ret.append([img_id[idx], loc_[0] * wtot, \
#                                 loc_[1] * htot,
#                                 (loc_[2] - loc_[0]) * wtot,
#                                 (loc_[3] - loc_[1]) * htot,
#                                 prob_,
#                                 inv_map[label_]])
#
#     # Now we have all predictions from this rank, gather them all together
#     # if necessary
#     ret = np.array(ret).astype(np.float32)
#
#
#     final_results = ret
#
#     cocoDt = cocoGt.loadRes(final_results)
#
#     # E = COCOeval(cocoGt, cocoDt, iouType='bbox')
#     # E.evaluate()
#     # E.accumulate()
#     #
#     return E.stats[0]  # Average Precision  (AP) @[ IoU=050:0.95 | area=   all | maxDets=100 ]