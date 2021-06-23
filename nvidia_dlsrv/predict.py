import argparse
import torch.utils.data
from dataset import MasksDataset
from eval import evaluate
from model import SSD300
import utils

# Parsing script arguments
parser = argparse.ArgumentParser(description='Process input')
parser.add_argument('input_folder', type=str, help='Input folder path, containing images')
args = parser.parse_args()

# Define device and checkpoint path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = r'/home/student/checkpoint_nvidia_ssd300_epoch=183.pth.tar'  # TODO YOTAM change
print(f"Evaluating data from path {args.input_folder}, checkpoint name {checkpoint}")

# Load model checkpoint that is to be evaluated
checkpoint = torch.load(checkpoint)
model = SSD300()
model.load_state_dict(checkpoint['state_dict'])
model = model.to(device)

# Switch to eval mode
model.eval()

# Load data
dataset = MasksDataset(data_folder=args.input_folder, split='test')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=24, shuffle=False,
                                         num_workers=1, pin_memory=True)
# Create boxes
boxes = utils.create_boxes()
encoder = utils.Encoder(boxes)

# Evaluate model on given data
evaluate(dataloader, model, encoder, save_csv="prediction.csv", verbose=True)
