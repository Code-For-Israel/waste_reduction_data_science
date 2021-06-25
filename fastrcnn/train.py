import time
import torch.optim
import torch.utils.data
from dataset import MasksDataset
from utils import AverageMeter, save_checkpoint
import constants
import pickle

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Label map
masks_labels = ('proper', 'not_porper')
label_map = {k: v + 1 for v, k in enumerate(masks_labels)}
label_map['background'] = 0
rev_label_map = {v: k for k, v in label_map.items()}  # Inverse mapping

distinct_colors = ['#e6194b', '#3cb44b', '#FFFFFF']
label_color_map = {k: distinct_colors[i] for i, k in enumerate(label_map.keys())}

# Model parameters
n_classes = len(label_map)  # number of different types of objects
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Learning parameters
batch_size = 42  # batch size TODO
workers = 4  # number of workers for loading data in the DataLoader TODO
print_freq = 200  # print training status every __ batches
lr = 1e-3  # learning rate TODO
weight_decay = 0  # weight decay TODO 5e-4
# clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32) -
# you will recognize it by a sorting error in the MuliBox loss calculation
grad_clip = None

checkpoint = ''  # '/home/student/checkpoint_ssd300_epoch=7.pth.tar'
if checkpoint:
    start_epoch = int(checkpoint.split('=')[-1].split('.')[0])
else:
    start_epoch = 0


def collate_fn(batch):
    return tuple(zip(*batch))


def main():
    """
    Training.
    """
    global label_map, epoch, decay_lr_at, checkpoint, device

    mean = [0.5244, 0.4904, 0.4781]
    std = [0.2642, 0.2608, 0.2561]

    # Initialize model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False,
                                                                 pretrained_backbone=False,
                                                                 image_mean=mean,
                                                                 image_std=std,
                                                                 min_size=300,
                                                                 max_size=300).to(device)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, n_classes).to(device)

    if checkpoint:
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint['state_dict'])

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Custom dataloaders
    train_dataset = MasksDataset(data_folder=constants.TRAIN_IMG_PATH, split='train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=workers, pin_memory=True, collate_fn=collate_fn)
    test_dataset = MasksDataset(data_folder=constants.TEST_IMG_PATH, split='test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                              num_workers=workers, pin_memory=True, collate_fn=collate_fn)

    # Calculate total number of epochs to train and the epochs to decay learning rate at
    # (i.e. convert iterations to epochs)
    # To convert iterations to epochs, divide iterations by the number of iterations per epoch
    # The paper trains for 120,000 iterations with a batch size of 32, decays after 80,000 and 100,000 iterations
    epochs = 500  # TODO change
    train_losses = []
    # Epochs
    for epoch in range(start_epoch, epochs):
        # One epoch's training
        epoch_loss = train(train_loader=train_loader,
                           model=model,
                           optimizer=optimizer,
                           epoch=epoch)
        # Get the epoch loss and append to list
        train_losses.append(epoch_loss)
        # Save all the losses to pickled list
        with open('/mnt/ml-srv1/home/yotamm/facemask_obj_detect/train_losses_list.pkl', 'wb') as f:
            pickle.dump(train_losses, f)

        # Save checkpoint
        save_checkpoint(epoch, model)

        # Load checkpoint TODO
        # checkpoint = torch.load('checkpoint_fasterrcnn_epoch=1.pth.tar')
        # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False,
        #                                                              pretrained_backbone=False,
        #                                                              image_mean=mean,
        #                                                              image_std=std,
        #                                                              min_size=300,
        #                                                              max_size=300).to(device)
        # in_features = model.roi_heads.box_predictor.cls_score.in_features
        # model.roi_heads.box_predictor = FastRCNNPredictor(in_features, n_classes).to(device)
        # model.load_state_dict(checkpoint['state_dict'])
        #
        # print('success')  # TODO

        # Evaluate test set
        # TODO eval


def train(train_loader, model, optimizer, epoch):
    """
    One epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses_meter = AverageMeter()  # loss

    start = time.time()

    # Batches
    for i, (images, targets) in enumerate(train_loader):
        data_time.update(time.time() - start)
        # Move to default device
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward prop
        loss_dict = model(images, targets)

        # Loss
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        # Backward prop.
        optimizer.zero_grad()
        losses.backward()

        # Update model
        optimizer.step()

        losses_meter.update(loss_value, len(images))
        batch_time.update(time.time() - start)
        start = time.time()

        # Print status
        if i % print_freq == 0 or i == len(train_loader) - 1:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses_meter))

    return losses_meter.avg


# TODO
#  1. Add early stopping based on test metrics / loss

if __name__ == '__main__':
    main()
