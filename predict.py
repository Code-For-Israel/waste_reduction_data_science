import argparse
import torch.utils.data
from dataset import MasksDataset
from model import get_fasterrcnn_resnet50_fpn
from eval import evaluate
import torch.backends.cudnn as cudnn
from dataset import collate_fn
import os
import gdown

cudnn.benchmark = True

# Parsing script arguments
parser = argparse.ArgumentParser(description='Process input')
parser.add_argument('input_folder', type=str, help='Input folder path, containing images')
args = parser.parse_args()

# Define device and checkpoint path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mean = [0.5244, 0.4904, 0.4781]
std = [0.2642, 0.2608, 0.2561]

print('Downloading model weights ...')
module_path = os.path.dirname(os.path.realpath(__file__))
gdrive_file_id = '1S8kCRrSA__mCI4Z0SPVuWNRty-iVlixS'  # TODO change to the best model

url = f'https://drive.google.com/uc?id={gdrive_file_id}'
weights_path = os.path.join(module_path, 'faster_rcnn.pth.tar')
gdown.download(url, weights_path, quiet=False)

print('Loading model ...')
model = get_fasterrcnn_resnet50_fpn()
checkpoint = torch.load(weights_path)
model.load_state_dict(checkpoint['state_dict'])

print('Loading data ...')
dataset = MasksDataset(data_folder=args.input_folder, split='test')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=5, shuffle=False,
                                         num_workers=1, pin_memory=False, collate_fn=collate_fn)

# Evaluate model on given data
print(f"Evaluating data from path {args.input_folder}")
evaluate(dataloader, model, save_csv="prediction.csv", verbose=True)
