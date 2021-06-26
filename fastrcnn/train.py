import time
import torch.optim
import torch.utils.data
from dataset import MasksDataset
from utils import AverageMeter, save_checkpoint
import constants
import pickle
from eval import evaluate
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
batch_size = 50  # batch size TODO
workers = 4  # number of workers for loading data in the DataLoader TODO
print_freq = 20  # print training status every __ batches
lr = 1e-3  # learning rate TODO
weight_decay = 0  # weight decay TODO 5e-4
# clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32) -
# you will recognize it by a sorting error in the MuliBox loss calculation
grad_clip = None


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
                                                                 min_size=224,
                                                                 max_size=224).to(device)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, n_classes).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Custom dataloaders
    train_dataset = MasksDataset(data_folder=constants.TRAIN_IMG_PATH, split='train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=workers, pin_memory=True, collate_fn=collate_fn)
    test_dataset = MasksDataset(data_folder=constants.TEST_IMG_PATH, split='test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                              num_workers=workers, pin_memory=True, collate_fn=collate_fn)

    # set split = test to avoid augmentations
    unshuffled_train_dataset = MasksDataset(data_folder=constants.TRAIN_IMG_PATH, split='test')
    unshuffled_train_loader = torch.utils.data.DataLoader(unshuffled_train_dataset, batch_size=batch_size,
                                                          shuffle=False, num_workers=workers, pin_memory=True,
                                                          collate_fn=collate_fn)

    # Calculate total number of epochs to train and the epochs to decay learning rate at
    # (i.e. convert iterations to epochs)
    # To convert iterations to epochs, divide iterations by the number of iterations per epoch
    # The paper trains for 120,000 iterations with a batch size of 32, decays after 80,000 and 100,000 iterations
    epochs = 500  # TODO change
    metrics = dict(train_loss=[], train_iou=[], train_accuracy=[],
                   test_loss=[], test_iou=[], test_accuracy=[])
    # Epochs
    for epoch in range(epochs):
        # One epoch's training
        train_loss = train(train_loader=train_loader,
                           model=model,
                           optimizer=optimizer,
                           epoch=epoch)

        # Save checkpoint
        save_checkpoint(epoch, model)

        # Evaluate train set
        train_mean_accuracy, train_mean_iou = evaluate(unshuffled_train_loader, model)
        print(f'Train IoU = {round(float(train_mean_iou), 4)}, Accuracy = {round(float(train_mean_accuracy), 4)}')

        # Test loss
        test_loss = get_test_loss(test_loader, model)

        # Evaluate test set
        test_mean_accuracy, test_mean_iou = evaluate(test_loader, model)
        print(f'Test IoU = {round(float(test_mean_iou), 4)}, Accuracy = {round(float(test_mean_accuracy), 4)}')

        # Populate dict
        metrics['train_loss'].append(train_loss)
        metrics['train_iou'].append(train_mean_iou)
        metrics['train_accuracy'].append(train_mean_accuracy)
        metrics['test_loss'].append(test_loss)
        metrics['test_iou'].append(test_mean_iou)
        metrics['test_accuracy'].append(test_mean_accuracy)

        # Save all the losses to pickled list
        with open('metrics.pkl', 'wb') as f:
            pickle.dump(metrics, f)


def get_test_loss(test_loader, model):
    losses_meter = AverageMeter()  # loss
    model.train()
    with torch.no_grad():
        for i, (images, targets) in enumerate(test_loader):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward prop
            loss_dict = model(images, targets)

            # Loss
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            losses_meter.update(loss_value, len(images))
    return losses_meter.avg


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
        if (i % print_freq == 0 or i == len(train_loader) - 1) and i != 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses_meter))

    return losses_meter.avg


if __name__ == '__main__':
    main()
