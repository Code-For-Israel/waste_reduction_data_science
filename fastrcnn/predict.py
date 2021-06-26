import argparse
import torch.utils.data
from dataset import MasksDataset
from model import get_fasterrcnn_resnet50_fpn
from eval import evaluate


def collate_fn(batch):
    return tuple(zip(*batch))


# Parsing script arguments
parser = argparse.ArgumentParser(description='Process input')
parser.add_argument('input_folder', type=str, help='Input folder path, containing images')
args = parser.parse_args()

# Define device and checkpoint path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mean = [0.5244, 0.4904, 0.4781]
std = [0.2642, 0.2608, 0.2561]

checkpoint = torch.load('/home/student/facemask_obj_detect/fastrcnn/checkpoint_fasterrcnn_epoch=4.pth.tar')  # TODO
model = get_fasterrcnn_resnet50_fpn()
model.load_state_dict(checkpoint['state_dict'])

# Load data
dataset = MasksDataset(data_folder=args.input_folder, split='test')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=20, shuffle=False,
                                         num_workers=1, pin_memory=False, collate_fn=collate_fn)

# Evaluate model on given data
print(f"Evaluating data from path {args.input_folder}")
evaluate(dataloader, model, save_csv="prediction.csv", verbose=True)
