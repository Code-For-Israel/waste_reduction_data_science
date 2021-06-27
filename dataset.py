from torch.utils.data import Dataset
import json
import os
from PIL import Image
from utils import *
import constants
import concurrent.futures
import torchvision.transforms.functional as FT
import random


def collate_fn(batch):
    return tuple(zip(*batch))


class MasksDataset(Dataset):
    """
    call example: MasksDataset(data_folder=constants.TRAIN_IMG_PATH, split='train')
    """

    def __init__(self, data_folder, split):
        self.split = split.upper()
        assert self.split in {'TRAIN', 'TEST'}

        self.data_folder = data_folder

        # Read data file names
        self.images = sorted(os.listdir(data_folder))
        if self.split == 'TRAIN':
            # exclude problematic images with width or heigh equal to 0
            self.paths_to_exclude = []
            for path in self.images:
                image_id, bbox, proper_mask = path.strip(".jpg").split("__")
                x_min, y_min, w, h = json.loads(bbox)  # convert string bbox to list of integers
                if w <= 0 or h <= 0:
                    self.paths_to_exclude.append(path)
            self.images = [path for path in self.images if path not in self.paths_to_exclude]

        # Load data to RAM using multiprocess
        self.loaded_imgs = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.load_single_img, path) for path in self.images]
            self.loaded_imgs = [fut.result() for fut in futures]
        self.loaded_imgs = sorted(self.loaded_imgs, key=lambda x: x[0])  # sort the images to reproduce results
        print(f"Finished loading {self.split} set to memory - total of {len(self.loaded_imgs)} images")

        # Store images sizes
        self.sizes = []
        for path in self.images:
            image = Image.open(os.path.join(self.data_folder, path), mode='r').convert('RGB')
            self.sizes.append(torch.FloatTensor([image.width, image.height, image.width, image.height]).unsqueeze(0))

    def __getitem__(self, i):
        mean = [0.5244, 0.4904, 0.4781]

        # MaskDataset train set mean and std
        image_id, image, box, label = self.loaded_imgs[i]  # str, PIL, tensor, tensor

        # Apply transformations and augmentations
        image, box, label = image.copy(), box.clone(), label.clone()
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
                    image, box = expand(image, box, filler=mean)

                # Randomly crop image (zoom in)
                if random.random() < 0.5:
                    image, box, label = random_crop(image, box, label)

                # Convert Torch tensor to PIL image
                image = FT.to_pil_image(image)

                # Flip image with a 50% chance
                if random.random() < 0.5:
                    image, box = flip(image, box)

        # non-fractional for Fast-RCNN
        image, box = resize(image, box, dims=(224, 224), return_percent_coords=False)  # PIL, tensor
        box = box.clamp(0., 224.)

        # Convert PIL image to Torch tensor
        image = FT.to_tensor(image)

        # No normalize for Fast-RCNN

        area = (box[:, 3] - box[:, 1]) * (box[:, 2] - box[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)

        target = dict(boxes=box,
                      labels=label,
                      image_id=torch.tensor([torch.tensor(int(image_id))]),
                      area=area,
                      iscrowd=torch.zeros_like(label, dtype=torch.int64))

        return image, target  # image is a tensor in [0, 1] (aka pixels divided by 255)

    def __len__(self):
        return len(self.images)

    def load_single_img(self, path):
        image_id, bbox, proper_mask = path.strip(".jpg").split("__")
        x_min, y_min, w, h = json.loads(bbox)  # convert string bbox to list of integers

        # it is promised that test set will not include non-positive w,h
        if (w <= 0 or h <= 0) and self.split == 'TEST':
            w = 1 if w <= 0 else w
            h = 1 if h <= 0 else h

        bbox = [x_min, y_min, x_min + w, y_min + h]  # [x_min, y_min, x_max, y_max]
        proper_mask = [1] if proper_mask.lower() == "true" else [2]

        # Read image
        image = Image.open(os.path.join(self.data_folder, path), mode='r').convert('RGB')

        box = torch.FloatTensor([bbox])  # (1, 4)
        label = torch.LongTensor(proper_mask)  # (1)

        return image_id, image, box, label  # str, PIL, tensor, tensor


if __name__ == '__main__':
    # check MasksDataset class
    # train
    dataset = MasksDataset(data_folder=constants.TRAIN_IMG_PATH, split='train')
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=20, shuffle=True, collate_fn=collate_fn)
    images, targets = next(iter(train_loader))

    # test
    dataset = MasksDataset(data_folder=constants.TEST_IMG_PATH, split='test')
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=20, shuffle=False, collate_fn=collate_fn)
