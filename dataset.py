import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from utils import transform
import constants


class MasksDataset(Dataset):
    """
    call example: MasksDataset(data_folder=constants.TRAIN_IMG_PATH, split='train')
    """

    def __init__(self, data_folder, split):
        self.split = split.upper()
        assert self.split in {'TRAIN', 'TEST'}

        self.data_folder = data_folder

        # Read data file names
        self.images = os.listdir(data_folder)

    def __getitem__(self, i):
        image_id, bbox, proper_mask = self.images[i].strip(".jpg").split("__")
        x_min, y_min, w, h = json.loads(bbox)  # convert string bbox to list of integers
        bbox = [x_min, y_min, x_min + w, y_min + h]  # [x_min, y_min, x_max, y_max]
        proper_mask = 1 if proper_mask.lower() == "true" else 0

        # Read image
        image = Image.open(os.path.join(self.data_folder, self.images[i]), mode='r')
        image = image.convert('RGB')

        box = torch.FloatTensor(bbox)  # (n_objects, 4) TODO YOTAM change comment
        label = torch.LongTensor(proper_mask)  # (n_objects) TODO YOTAM change comment

        # Apply transformations
        image, box, label = transform(image, box, label, split=self.split)

        return image, box, label

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    # check MasksDataset class
    dataset = MasksDataset(data_folder=constants.TRAIN_IMG_PATH, split='train')
    next(iter(dataset))
