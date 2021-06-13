import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from utils import transform
import constants


# TODO YOTAM: possibly load all to RAM to save some time

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

        # Store images sizes
        self.sizes = []
        for path in self.images:
            image = Image.open(os.path.join(self.data_folder, path), mode='r').convert('RGB')
            self.sizes.append(torch.FloatTensor([image.width, image.height, image.width, image.height]).unsqueeze(0))

    def __getitem__(self, i):
        image_id, bbox, proper_mask = self.images[i].strip(".jpg").split("__")
        x_min, y_min, w, h = json.loads(bbox)  # convert string bbox to list of integers
        bbox = [x_min, y_min, x_min + w, y_min + h]  # [x_min, y_min, x_max, y_max]
        proper_mask = [1] if proper_mask.lower() == "true" else [2]

        # Read image
        image = Image.open(os.path.join(self.data_folder, self.images[i]), mode='r')
        image = image.convert('RGB')

        box = torch.FloatTensor(bbox)  # (1, 4)
        label = torch.LongTensor(proper_mask)  # (1)

        # Apply transformations
        image, box, label = transform(image, box, label, split=self.split)

        return image, box, label

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    # check MasksDataset class
    dataset = MasksDataset(data_folder=constants.TRAIN_IMG_PATH, split='train')
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True,
                                               num_workers=4, pin_memory=True)
    (images, boxes, labels) = next(iter(train_loader))
