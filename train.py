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

# Learning parameters TODO Choose parameters
batch_size = 10  # batch size
workers = 1  # number of workers for loading data in the DataLoader
print_freq = 1  # print training status every __ batches
lr = 1e-5  # learning rate

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
    train_dataset = TrucksDataset(data_folder=constants.TRAIN_DIRECTORY_PATH, split='train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=workers, pin_memory=True, collate_fn=collate_fn)
    test_dataset = TrucksDataset(data_folder=constants.TEST_DIRECTORY_PATH, split='test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                              num_workers=workers, pin_memory=True, collate_fn=collate_fn)

    # set split = test to avoid augmentations
    unshuffled_train_dataset = TrucksDataset(data_folder=constants.TRAIN_DIRECTORY_PATH, split='test')
    unshuffled_train_loader = torch.utils.data.DataLoader(unshuffled_train_dataset, batch_size=batch_size,
                                                          shuffle=False, num_workers=workers, pin_memory=True,
                                                          collate_fn=collate_fn)

    epochs = 200  # TODO More epochs
    metrics = dict(train_loss=[], train_APs=[], train_mAP=[],
                   test_APs=[], test_mAP=[])
    # Epochs
    for epoch in range(epochs):
        # One epoch's training
        train_loss = train(train_loader=train_loader,
                           model=model,
                           optimizer=optimizer,
                           epoch=epoch)

        # Save checkpoint - TODO UNCOMMENT
        save_checkpoint(epoch, model)

        # Evaluate train set
        train_APs, train_mAP = evaluate(unshuffled_train_loader, model)
        # TODO UNCOMMENT
        print(f'[Train set] Class-wise average precisions: {train_APs}')
        print(f'[Train set] Mean Average Precision (mAP): {round(train_mAP, 3)}')

        # Evaluate test set
        test_APs, test_mAP = evaluate(test_loader, model)
        # TODO UNCOMMENT
        print(f'[Test set] Class-wise average precisions: {test_APs}')
        print(f'[Test set] Mean Average Precision (mAP): {round(test_mAP, 3)}')

        # Populate dict
        metrics['train_loss'].append(train_loss)
        metrics['train_APs'].append(train_APs)
        metrics['train_mAP'].append(train_mAP)
        metrics['test_APs'].append(test_APs)
        metrics['test_mAP'].append(test_mAP)

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

        print('loss_value ', loss_value)  # TODO Del

        # Backward prop.
        optimizer.zero_grad()
        losses.backward()

        # TODO DEL /  COMMENT
        print('model params contain at least 1 nan: ',
              any([torch.isnan(p).any() for p in model.parameters()]))

        # Clipping - TODO needed?
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=2)

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
