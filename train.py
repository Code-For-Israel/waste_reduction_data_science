import time
import torch.optim
import torch.utils.data
import torch.backends.cudnn as cudnn
from dataset import TrucksDataset
from utils import AverageMeter, save_checkpoint
import constants
import pickle
from eval import evaluate
from model import get_fasterrcnn_resnet50_fpn
from dataset import collate_fn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Learning parameters
batch_size = 42  # batch size (We trained with 42)
workers = 4  # number of workers for loading data in the DataLoader
print_freq = 10  # print training status every __ batches
lr = 1e-3  # learning rate

cudnn.benchmark = True


def main():
    """
    Training.
    """
    global device

    # Initialize model
    model = get_fasterrcnn_resnet50_fpn()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Custom dataloaders
    train_dataset = TrucksDataset(data_folder=constants.TRAIN_IMG_PATH, split='train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=workers, pin_memory=True, collate_fn=collate_fn)
    test_dataset = TrucksDataset(data_folder=constants.TEST_IMG_PATH, split='test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                              num_workers=workers, pin_memory=True, collate_fn=collate_fn)

    # set split = test to avoid augmentations
    unshuffled_train_dataset = TrucksDataset(data_folder=constants.TRAIN_IMG_PATH, split='test')
    unshuffled_train_loader = torch.utils.data.DataLoader(unshuffled_train_dataset, batch_size=batch_size,
                                                          shuffle=False, num_workers=workers, pin_memory=True,
                                                          collate_fn=collate_fn)

    epochs = 100
    metrics = dict(train_loss=[], train_iou=[], train_accuracy=[],
                   test_iou=[], test_accuracy=[])
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

        # Evaluate test set
        test_mean_accuracy, test_mean_iou = evaluate(test_loader, model)
        print(f'Test IoU = {round(float(test_mean_iou), 4)}, Accuracy = {round(float(test_mean_accuracy), 4)}')

        # Populate dict
        metrics['train_loss'].append(train_loss)
        metrics['train_iou'].append(train_mean_iou)
        metrics['train_accuracy'].append(train_mean_accuracy)
        metrics['test_iou'].append(test_mean_iou)
        metrics['test_accuracy'].append(test_mean_accuracy)

        # Save all the losses to pickled list
        with open('metrics.pkl', 'wb') as f:
            pickle.dump(metrics, f)

        torch.cuda.empty_cache()


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

        # Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=2)

        # Print gradients norms to know what max_norm to give
        # max_norm = 0
        # for name, param in model.named_parameters():
        #     norm = param.grad.norm(2)
        #     # print(name, norm)
        #     if norm > max_norm:
        #         max_norm = norm
        # print(f'MAX NORM = {max_norm}')

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
        del loss_dict, losses, images, targets  # free some memory since their histories may be stored
    torch.cuda.empty_cache()
    return losses_meter.avg


if __name__ == '__main__':
    main()
