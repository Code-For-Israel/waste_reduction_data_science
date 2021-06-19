import os
import argparse
import torch
import torch.utils.data
from dataset import MasksDataset
from eval import evaluate
from model import SSD300

# Parsing script arguments
parser = argparse.ArgumentParser(description='Process input')
parser.add_argument('input_folder', type=str, help='Input folder path, containing images')
args = parser.parse_args()

# Define device and checkpoint path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = 'checkpoint_ssd300_epoch=1.pth.tar'  # TODO YOTAM change

# Label map
masks_labels = ('proper', 'not_porper')
label_map = {k: v + 1 for v, k in enumerate(masks_labels)}
label_map['background'] = 0
rev_label_map = {v: k for k, v in label_map.items()}  # Inverse mapping

distinct_colors = ['#e6194b', '#3cb44b', '#FFFFFF']
label_color_map = {k: distinct_colors[i] for i, k in enumerate(label_map.keys())}

# Model parameters
n_classes = len(label_map)  # number of different types of objects

# Load model checkpoint that is to be evaluated
checkpoint = torch.load(checkpoint)
model = SSD300(n_classes=n_classes)
model.load_state_dict(checkpoint['state_dict'])
model = model.to(device)

# Switch to eval mode
model.eval()

# Load data
dataset = MasksDataset(data_folder=args.input_folder, split='test')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False,
                                         num_workers=4, pin_memory=True)

# Evaluate model on given data
evaluate(dataloader, model, save_csv="prediction.csv", verbose=True)