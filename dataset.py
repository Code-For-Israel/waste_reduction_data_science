import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from utils import transform
import constants
import concurrent.futures


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

        # Load data to RAM using multiprocess
        self.loaded_imgs = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.load_single_img, path) for path in self.images]
            self.loaded_imgs = [fut.result() for fut in futures]
        self.loaded_imgs = sorted(self.loaded_imgs, key=lambda x: x[0])  # sort the images to reproduce results
        print(f"Finished loading {self.split} set to memory")

        # Store images sizes
        self.sizes = []
        for path in self.images:
            image = Image.open(os.path.join(self.data_folder, path), mode='r').convert('RGB')
            self.sizes.append(torch.FloatTensor([image.width, image.height, image.width, image.height]).unsqueeze(0))

    def __getitem__(self, i):
        sample = self.loaded_imgs[i]
        # TODO YOTAM if split == TRAIN make transformations (crop, flip ...)
        return sample[1], sample[2], sample[3]  # image, box, label

    def __len__(self):
        return len(self.images)

    def load_single_img(self, path):
        image_id, bbox, proper_mask = path.strip(".jpg").split("__")
        x_min, y_min, w, h = json.loads(bbox)  # convert string bbox to list of integers
        bbox = [x_min, y_min, x_min + w, y_min + h]  # [x_min, y_min, x_max, y_max]
        proper_mask = [1] if proper_mask.lower() == "true" else [2]

        # Read image
        image = Image.open(os.path.join(self.data_folder, path), mode='r').convert('RGB')
        image = image.convert('RGB')

        box = torch.FloatTensor(bbox)  # (1, 4)
        label = torch.LongTensor(proper_mask)  # (1)

        # Apply transformations
        image, box, label = transform(image, box, label, split=self.split)

        return image_id, image, box, label


if __name__ == '__main__':
    # check MasksDataset class
    # total of ~22GB RAM are needed
    # train
    dataset = MasksDataset(data_folder=constants.TRAIN_IMG_PATH, split='train')
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True,
                                               num_workers=4, pin_memory=True)
    (images, boxes, labels) = next(iter(train_loader))
    print(dataset.loaded_imgs[-1])

    # test
    dataset = MasksDataset(data_folder=constants.TEST_IMG_PATH, split='test')
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False,
                                              num_workers=4, pin_memory=True)
    print(dataset.loaded_imgs[-1])
