import os
import argparse
import torch
import torch.utils.data
from dataset import MasksDataset
from eval import evaluate

# Parsing script arguments
parser = argparse.ArgumentParser(description='Process input')
parser.add_argument('input_folder', type=str, help='Input folder path, containing images')
args = parser.parse_args()

#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = './checkpoint_ssd300.pth.tar'  # TODO YOTAM change

# Load model checkpoint that is to be evaluated
checkpoint = torch.load(checkpoint)
model = checkpoint['model']
model = model.to(device)

# Switch to eval mode
model.eval()

# Load data
dataset = MasksDataset(data_folder=args.input_folder, split='test')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False,
                                         num_workers=4, pin_memory=True)

# Evaluate model on given data
evaluate(dataloader, model, save_csv="prediction.csv", verbose=True)
