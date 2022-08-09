import torch
from utils import calculate_mAP


def evaluate(loader, model, verbose=False):
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
                if len(det_boxes):  # TODO DEL prints
                    print('det_boxes ', det_boxes)
                    print('true_boxes ', targets[k]['boxes']),

                true_boxes.append(targets[k]['boxes'])
                true_labels.append(targets[k]['labels'])
                # TODO What should be the difficulties
                true_difficulties.append(torch.zeros_like(targets[k]['labels']))  # just for using `calculate_mAP()`

            del images, res, det_boxes, det_labels, det_scores

        # Calculate mAP
        APs, mAP = calculate_mAP(detected_boxes, detected_labels, detected_scores,
                                 true_boxes, true_labels, true_difficulties)

    # Print AP for each class
    if verbose:
        print(f'Class-wise average precisions: {APs}')
        print(f'Mean Average Precision (mAP): {round(mAP, 3)}')

    del detected_boxes, detected_labels, detected_scores, true_boxes, true_labels, true_difficulties
    torch.cuda.empty_cache()

    return APs, mAP
