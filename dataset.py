from typing import Tuple
from torch.utils.data import Dataset
import os
from PIL import Image
from utils import *
import constants
import concurrent.futures
import torchvision.transforms.functional as FT
import random


def collate_fn(batch):
    return tuple(zip(*batch))


class TrucksDataset(Dataset):
    """
    call example: TrucksDataset(data_folder=constants.TRAIN_IMG_PATH, split='train')
    """

    def __init__(self, data_folder: str, split: str):
        self.split = split.upper()
        assert self.split in {'TRAIN', 'TEST'}

        self.data_folder = data_folder  # Absolute path to directory with .jpg and .txt files

        # Read data file names
        # self.filenames = ['img00010000351_06_02_2022T15_16_52', 'img00010000352_06_02_2022T15_29_06', ...]
        self.filenames = [filename for filename in sorted(os.listdir(data_folder)) if
                          '.txt' in filename or '.jpg' in filename]
        self.filenames = sorted(set([filename.replace('.txt', '').replace('.jpg', '') for filename in self.filenames]))

        # Load data to RAM using multiprocess
        self.data = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.load_single_img_and_data, filename) for filename in self.filenames]
            self.data = [fut.result() for fut in futures]
        self.data = sorted(self.data, key=lambda x: x[0])  # sort the filenames to reproduce results
        print(f"Finished loading {self.split} set to memory - total of {len(self.data)} images")

        # Store filenames sizes
        self.sizes = []
        for filename_without_extension, image, boxes, labels in self.data:
            self.sizes.append(torch.FloatTensor([image.width, image.height, image.width, image.height]).unsqueeze(0))

    def __getitem__(self, i):
        # TrucksDataset mean
        mean = [0.5244, 0.4904, 0.4781]

        # MaskDataset train set mean and std
        filename_without_extension, image, boxes, labels = self.data[i]  # str, PIL, tensor, tensor
        image_id = int(filename_without_extension.split('_')[0].replace('img', ''))

        # Apply transformations and augmentations
        image, boxes, labels = image.copy(), boxes.clone(), labels.clone()
        if self.split == 'TRAIN':
            if random.random() < 0.8:  # with probability of 80% try augmentations
                # A series of photometric distortions in random order, each with 50% chance of occurrence, as in Caffe repo
                if random.random() < 0.5:
                    image = photometric_distort(image)

                # Convert PIL image to Torch tensor
                image = FT.to_tensor(image)

                # Expand image (zoom out) with a 50% chance - helpful for training detection of small objects
                # Fill surrounding space with the mean
                if random.random() < 0.5:
                    image, boxes = expand(image, boxes, filler=mean)

                # Randomly crop image (zoom in)
                if random.random() < 0.5:
                    image, boxes, labels = random_crop(image, boxes, labels)

                # Convert Torch tensor to PIL image
                image = FT.to_pil_image(image)

                # Flip image with a 50% chance
                if random.random() < 0.5:
                    image, boxes = flip(image, boxes)

        # non-fractional for Fast-RCNN
        image, boxes = resize(image, boxes, dims=(224, 224), return_percent_coords=False)  # PIL, tensor
        boxes = boxes.clamp(0., 224.)

        # Convert PIL image to Torch tensor
        image = FT.to_tensor(image)

        # No normalize for Fast-RCNN

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)

        target = dict(boxes=boxes,
                      labels=labels,
                      image_id=torch.tensor([torch.tensor(int(image_id))]),
                      area=area,
                      iscrowd=torch.zeros_like(labels, dtype=torch.int64))

        return image, target  # image is a tensor in [0, 1] (aka pixels divided by 255)

    def __len__(self):
        return len(self.data)

    def extract_bboxes_and_labels_from_annotations_txt(self, annotations_path) -> Tuple:
        """
        annotations_path: str, path to yolo1.1 like annotations txt file
        with each row as "label center_x center_y width height" e.g.
        3 0.628125 0.8166666666666667 0.115625 0.3

        return tuple of two lists
        """
        boxes, labels = list(), list()
        annotations = [row.strip().split(' ') for row in
                       open(os.path.join(self.data_folder, annotations_path)).readlines()]
        for annotation in annotations:
            label, center_x, center_y, width, height = annotation
            label = int(label)
            center_x, center_y, width, height = float(center_x), float(center_y), float(width), float(height)
            labels.append(label)
            boxes.append([center_x, center_y, width, height])

        return boxes, labels

    def load_single_img_and_data(self, filename_without_extension: str):
        assert '.txt' not in filename_without_extension and '.jpg' not in filename_without_extension
        image_path = filename_without_extension + '.jpg'
        annotations_path = filename_without_extension + '.txt'

        boxes, labels = self.extract_bboxes_and_labels_from_annotations_txt(annotations_path)

        boxes = torch.FloatTensor(boxes)  # shape (n_boxes, 4), each box is [center_x, center_y, width, height]
        boxes = cxcy_to_xy(boxes)  # shape (n_boxes, 4), each box is
        labels = torch.LongTensor(labels)  # shape (n_boxes)

        # Read image
        image = Image.open(os.path.join(self.data_folder, image_path), mode='r').convert('RGB')

        return filename_without_extension, image, boxes, labels  # str, PIL, tensor, tensor


if __name__ == '__main__':
    # check TrucksDataset class
    # train
    dataset = TrucksDataset(data_folder=constants.TRAIN_IMG_PATH, split='train')
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=20, shuffle=True, collate_fn=collate_fn)
    images, targets = next(iter(train_loader))

    # test
    dataset = TrucksDataset(data_folder=constants.TEST_IMG_PATH, split='test')
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=20, shuffle=False, collate_fn=collate_fn)
