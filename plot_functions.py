import os
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import patches
from utils import calc_iou
from matplotlib.ticker import StrMethodFormatter

# TODO Find better way to plot? with utils.py from sgrvinod repo

np.random.seed(42)
image_dir = "/home/student/test"


def parse_images_and_bboxes(image_dir):
    """
    Parse a directory with filenames.
    :param image_dir: Path to directory with filenames.
    :return: A list with (filename, image_id, bbox, proper_mask) for every image in the image_dir.
    """
    example_filenames = os.listdir(image_dir)
    data = []
    for filename in example_filenames:
        image_id, bbox, proper_mask = filename.strip(".jpg").split("__")
        bbox = json.loads(bbox)
        proper_mask = True if proper_mask.lower() == "true" else False
        data.append((filename, image_id, bbox, proper_mask))
    return data


def show_images_and_bboxes(data, image_dir):
    """
    Plot filenames with bounding boxes. Predicts random bounding boxes and computes IoU.
    :param data: Iterable with (filename, image_id, bbox, proper_mask) structure.
    :param image_dir: Path to directory with filenames.
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
        # x2, y2, w2, h2 = random_bbox_predict(bbox)
        # Calculate IoU
        # iou = calc_iou(bbox, (x2, y2, w2, h2))
        # Plot image and bboxes
        fig, ax = plt.subplots()
        ax.imshow(im)
        rect = patches.Rectangle((x1, y1), w1, h1,
                                 linewidth=2, edgecolor='g', facecolor='none', label='ground-truth')
        ax.add_patch(rect)
        # rect = patches.Rectangle((x2, y2), w2, h2,
        #                          linewidth=2, edgecolor='b', facecolor='none', label='predicted')
        # ax.add_patch(rect)
        # fig.suptitle(f"proper_mask={proper_mask}, IoU={iou:.2f}")
        fig.suptitle(f"proper_mask={proper_mask}")
        ax.axis('off')
        fig.legend()
        plt.show()


def show_images_and_bboxes_from_predictions_df(path):
    """
    Plot filenames with bounding boxes. Predicts random bounding boxes and computes IoU.
    :param path: path to final predictions.csv file, after evaluation of a model
    :return: None
    """
    df = pd.read_csv(path)
    df['true_box'] = df['filename'].apply(lambda x: [float(i) for i in x.split('__')[1][1:-1].split(',')])
    df['true_label'] = df['filename'].apply(lambda x: x.split('__')[2][:-4])
    df['iou'] = df.apply(lambda sample: calc_iou(sample.true_box, (sample.x, sample.y, sample.w, sample.h)), axis=1)
    df = df.sort_values(by=['iou'])

    confusion_matrix = pd.crosstab(df['true_label'], df['proper_mask'], rownames=['Actual'], colnames=['Predicted'])
    print(confusion_matrix.values)

    plot_iou_hist(df)
    aux_plot_examples(df.head())
    aux_plot_examples(df.tail())


def plot_iou_hist(df):
    ax = df.hist(column='iou', bins=20, grid=False, figsize=(12, 8), color='#86bf91', zorder=2,
                 rwidth=0.9)

    ax = ax[0]
    for x in ax:

        # Despine
        x.spines['right'].set_visible(False)
        x.spines['top'].set_visible(False)
        x.spines['left'].set_visible(False)

        # Switch off ticks
        x.tick_params(bottom="off", top="off", labelbottom="on", left="off", right="off",
                      labelleft="on")

        # Draw horizontal axis lines
        vals = x.get_yticks()
        for tick in vals:
            x.axhline(y=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)

        # Remove title
        x.set_title("IOU Histogram")

        # Set x-axis label
        x.set_xlabel("IOU", labelpad=20, weight='bold', size=12)

        # Set y-axis label
        x.set_ylabel("Frequency", labelpad=20, weight='bold', size=12)

        # Format y-axis label
        x.yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))
    plt.show()


def aux_plot_examples(df):
    for index, row in df.iterrows():
        # Load image
        im = cv2.imread(os.path.join(image_dir, row.filename))
        # BGR to RGB
        im = im[:, :, ::-1]
        # Ground truth bbox
        x1, y1, w1, h1 = row.true_box
        # Predicted bbox
        x2, y2, w2, h2 = row.x, row.y, row.w, row.h
        # Calculate IoU
        iou = calc_iou((x1, y1, w1, h1), (x2, y2, w2, h2))
        # Plot image and bboxes
        fig, ax = plt.subplots()
        ax.imshow(im)
        rect = patches.Rectangle((x1, y1), w1, h1,
                                 linewidth=2, edgecolor='g', facecolor='none', label='ground-truth')
        ax.add_patch(rect)
        rect = patches.Rectangle((x2, y2), w2, h2,
                                 linewidth=2, edgecolor='b', facecolor='none', label='predicted')
        ax.add_patch(rect)
        fig.suptitle(f"True label={row.true_label}, Predicted label={row.proper_mask}, IoU={iou:.2f}")
        ax.axis('off')
        fig.legend(loc='lower right')
        plt.show()


def plot_one_metric(train_list, test_list, metric, mark_epoch=None):
    """
    Plot graph with train and test results.
    :param train_list: train results values.
    :param test_list: test results values.
    :param metric: string of the metric name.
    :return: None.
    """
    indices_list = [(1 + i) for i in range(len(train_list))]
    plt.figure(figsize=(12, 2))
    plt.plot(indices_list, train_list, '-', c="tab:blue", label=f'Train {metric}')
    plt.plot(indices_list, test_list, '-', c="tab:orange", label=f'Test {metric}')

    plt.plot(indices_list, train_list, 'o', color='tab:blue', markersize=4)
    plt.plot(indices_list, test_list, 'o', color='tab:orange', markersize=4)

    # Mark the chosen epoch
    if mark_epoch:
        plt.plot([mark_epoch], train_list[mark_epoch - 1], 'o', color='tab:red', markersize=6)
        plt.plot([mark_epoch], test_list[mark_epoch - 1], 'o', color='tab:red', markersize=6)

    plt.xticks(np.arange(1, len(indices_list) + 1, step=1), fontsize=10, rotation=90)
    plt.grid(linewidth=1)
    plt.title(f'Train and test {metric} values along epochs')
    plt.xlabel("Epochs")
    plt.ylabel(f'{metric}')
    plt.legend(loc='lower right' if metric != 'loss' else 'upper right')
    plt.show()


if __name__ == "__main__":
    # plot example filenames
    # data = parse_images_and_bboxes(image_dir)
    # show_images_and_bboxes(data, image_dir)

    # plot sample of predicted BBs along the true ones
    predictions_path = '/tmp/pycharm_project_388/prediction.csv'
    show_images_and_bboxes_from_predictions_df(predictions_path)

    # plot the metrics graphs
    # plot_one_metric(train_list=[0, 1, 2], test_list=[4, 5, 7], metric='loss')
    # import pickle
    #
    # exp1_metrics = pickle.load(open('exp1 metrics.pkl.txt', 'rb'))
    # plot_one_metric(train_list=exp1_metrics['train_loss'], test_list=exp1_metrics['test_loss'], metric='loss')
    # plot_one_metric(train_list=exp1_metrics['train_iou'], test_list=exp1_metrics['test_iou'], metric='iou')
    # plot_one_metric(train_list=exp1_metrics['train_accuracy'], test_list=exp1_metrics['test_accuracy'],
    #                 metric='accuracy')
