import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from utils import *
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
        self.images = sorted(os.listdir(data_folder))
        if self.split == 'TRAIN':
            # exclude problematic images with width or heigh equal to 0
            paths_to_exclude = []
            for path in self.images:
                image_id, bbox, proper_mask = path.strip(".jpg").split("__")
                x_min, y_min, w, h = json.loads(bbox)  # convert string bbox to list of integers
                if w <= 0 or h <= 0:
                    paths_to_exclude.append(path)
            self.images = [path for path in self.images if path not in paths_to_exclude]

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
        if self.split == 'TRAIN':
            # get sample from saved object
            image_id, image, box, label = self.loaded_imgs[i]  # str, PIL.image, tensor, tensor

            # copy sample to avoid making changes to original data
            new_image, new_box, new_label = image.copy(), box.clone(), label.clone()

            # Apply transformations and augmentations
            new_image, new_box, new_label = transform(new_image, new_box, new_label, split=self.split)
            return new_image, new_box, new_label
        else:
            # get sample from saved object
            image_id, image, box, label = self.loaded_imgs[i]  # str, tensor, tensor, tensor
            return image, box, label

    def __len__(self):
        return len(self.images)

    def load_single_img(self, path):
        image_id, bbox, proper_mask = path.strip(".jpg").split("__")
        x_min, y_min, w, h = json.loads(bbox)  # convert string bbox to list of integers
        bbox = [x_min, y_min, x_min + w, y_min + h]  # [x_min, y_min, x_max, y_max]
        bbox = [number if number != 0 else 1e-8 for number in bbox]  # to avoid inf in smooth_l1 in loss function
        proper_mask = [1] if proper_mask.lower() == "true" else [2]

        # Read image
        image = Image.open(os.path.join(self.data_folder, path), mode='r').convert('RGB')

        box = torch.FloatTensor([bbox])  # (1, 4)
        label = torch.LongTensor(proper_mask)  # (1)

        if self.split == 'TRAIN':
            return image_id, image, box, label  # str, PIL.image, tensor, tensor
        else:
            mean = [0.5244, 0.4904, 0.4781]
            std = [0.2642, 0.2608, 0.2561]

            # Resize image to (300, 300) - this also converts absolute boundary coordinates to their fractional form
            new_image, new_box = resize(image, box, dims=(300, 300))

            # Convert PIL image to Torch tensor
            new_image = FT.to_tensor(new_image)

            # Normalize by mean and standard deviation of ImageNet data that our base VGG was trained on
            new_image = FT.normalize(new_image, mean=mean, std=std)
            return image_id, new_image, new_box, label


def calculate_mean_std(path_to_images, size=300):
    """
    Calculate the mean ans std of image dataset
    :param path_to_images: (str) path to images folder
    """
    from torch.utils.data import Dataset
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from PIL import Image
    import os

    class CustomDataSet(Dataset):
        def __init__(self, main_dir, transform):
            self.main_dir = main_dir
            self.transform = transform
            self.all_imgs = os.listdir(main_dir)

        def __len__(self):
            return len(self.all_imgs)

        def __getitem__(self, idx):
            img_loc = os.path.join(self.main_dir, self.all_imgs[idx])
            image = Image.open(img_loc).convert("RGB")
            tensor_image = self.transform(image)
            return tensor_image

    dataset = CustomDataSet(path_to_images, transforms.Compose([transforms.Resize(size), transforms.ToTensor()]))

    loader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False)

    mean = 0.
    std = 0.
    nb_samples = 0.
    for data in loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    print('train pixel mean values', mean)
    print('train pixel std values', std)


if __name__ == '__main__':
    # check MasksDataset class
    # total of ~22GB RAM are needed
    # train
    dataset = MasksDataset(data_folder=constants.TRAIN_IMG_PATH, split='train')
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=24, shuffle=True)
    (images, boxes, labels) = next(iter(train_loader))

    # test
    dataset = MasksDataset(data_folder=constants.TEST_IMG_PATH, split='test')
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=24, shuffle=False)
