import time
import torch.optim
import torch.utils.data
import torch.backends.cudnn as cudnn
from dataset import MasksDataset
from utils import AverageMeter, save_checkpoint
import constants
import pickle
from eval import evaluate
from model import get_fasterrcnn_resnet50_fpn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Learning parameters
batch_size = 42  # batch size TODO
workers = 4  # number of workers for loading data in the DataLoader TODO
print_freq = 20  # print training status every __ batches
lr = 1e-3  # learning rate TODO
weight_decay = 0  # weight decay TODO 5e-4
# clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32) -
# you will recognize it by a sorting error in the MuliBox loss calculation
grad_clip = None
cudnn.benchmark = True


def collate_fn(batch):
    return tuple(zip(*batch))


def main():
    """
    Training.
    """
    global device

    # Initialize model
    model = get_fasterrcnn_resnet50_fpn()

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

        # Evaluate test set
        test_mean_accuracy, test_mean_iou = evaluate(test_loader, model)
        print(f'Test IoU = {round(float(test_mean_iou), 4)}, Accuracy = {round(float(test_mean_accuracy), 4)}')

        # Test loss
        test_loss = get_test_loss(test_loader, model)

        # Populate dict
        metrics['train_loss'].append(train_loss)
        metrics['train_iou'].append(train_mean_iou)
        metrics['train_accuracy'].append(train_mean_accuracy)
        metrics['test_loss'].append(test_loss)
        metrics['test_iou'].append(test_mean_iou)
        metrics['test_accuracy'].append(test_mean_accuracy)

        # Save all the losses to pickled list
        with open('exp2_metrics.pkl', 'wb') as f:
            pickle.dump(metrics, f)

        torch.cuda.empty_cache()


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
    del loss_dict, losses, images, targets  # free some memory since their histories may be stored
    torch.cuda.empty_cache()
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

        # Clipping TODO maybe smaller max_norm? (2?)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2)
        # printing gradients norms
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


# TODO
#  1. better most-left mechanism
#  2. try other learning rate
#  3. try weight_decay

# TODO EXP-1 commit:
#  https://github.com/yotammarton/facemask_obj_detect/tree/0da6c19dad696ce810fd3274cd26a1c539717681
#  METRICS:
"""
train_loss [3.9092, 0.2825, 0.2548, 0.2512, 0.2414, 0.2338, 0.4992, 0.301, 0.2705, 0.262, 0.2631, 0.2562, 0.2468, 0.2428, 0.2394, 0.2332, 0.2268, 0.2215, 0.2146, 0.2156, 0.213, 0.2112, 0.2101, 0.2081, 0.2075, 0.2045, 0.2061, 0.2032, 0.2002, 0.2051, 0.2012, 0.1944, 0.1977, 0.197, 0.1912, 0.1943, 0.1984, 0.1948, 0.1946, 0.1942, 0.1951, 0.1927, 0.1955, 0.1932, 0.1943, 0.1921, 0.1849]
train_iou [0.1205, 0.1303, 0.1515, 0.2202, 0.2568, 0.262, 0.0706, 0.1072, 0.1164, 0.1081, 0.1644, 0.2318, 0.2363, 0.2262, 0.2762, 0.2928, 0.3189, 0.2893, 0.3965, 0.3423, 0.4274, 0.337, 0.4017, 0.3692, 0.3914, 0.4493, 0.4309, 0.392, 0.376, 0.4493, 0.3996, 0.3359, 0.4671, 0.4174, 0.403, 0.4542, 0.4351, 0.3912, 0.439, 0.4227, 0.4554, 0.4893, 0.413, 0.446, 0.3844, 0.3688, 0.4519]
train_accuracy [0.5337, 0.5412, 0.5293, 0.5406, 0.5514, 0.5525, 0.4807, 0.5128, 0.5125, 0.4964, 0.5304, 0.5026, 0.5293, 0.5352, 0.5444, 0.5474, 0.553, 0.5555, 0.5493, 0.5474, 0.5713, 0.5574, 0.5759, 0.5734, 0.5822, 0.5396, 0.5822, 0.5704, 0.5662, 0.6242, 0.5342, 0.5581, 0.5691, 0.6068, 0.5768, 0.5454, 0.5828, 0.5778, 0.5661, 0.5654, 0.5538, 0.6191, 0.6051, 0.5867, 0.5619, 0.5988, 0.5644]
test_loss [0.2912, 0.2641, 0.2607, 0.2306, 0.2291, 0.2707, 0.4736, 0.2763, 0.2956, 0.2955, 0.2815, 0.2594, 0.2506, 0.2495, 0.2503, 0.2371, 0.2346, 0.225, 0.2273, 0.218, 0.2159, 0.2205, 0.2157, 0.2195, 0.1993, 0.2171, 0.2118, 0.2145, 0.1897, 0.2156, 0.1842, 0.223, 0.1859, 0.1881, 0.2121, 0.232, 0.2113, 0.2259, 0.2041, 0.1857, 0.1931, 0.1886, 0.2053, 0.1883, 0.1957, 0.2093, 0.2205]
test_iou [0.1223, 0.129, 0.1505, 0.2245, 0.2592, 0.2637, 0.0697, 0.1091, 0.1157, 0.1071, 0.1641, 0.2336, 0.2352, 0.2276, 0.2761, 0.2976, 0.3217, 0.2955, 0.4022, 0.3388, 0.429, 0.3335, 0.3998, 0.37, 0.395, 0.4507, 0.4331, 0.3871, 0.3757, 0.4495, 0.4019, 0.3396, 0.4649, 0.42, 0.4015, 0.4555, 0.4294, 0.3888, 0.4357, 0.4168, 0.4539, 0.4871, 0.4182, 0.4444, 0.3802, 0.3649, 0.4503]
test_accuracy [0.5325, 0.541, 0.5305, 0.544, 0.5397, 0.5592, 0.4738, 0.5085, 0.4948, 0.4822, 0.5262, 0.5242, 0.53, 0.5342, 0.5438, 0.5505, 0.5498, 0.554, 0.5585, 0.5472, 0.5608, 0.5535, 0.5702, 0.569, 0.5858, 0.538, 0.586, 0.5795, 0.5712, 0.635, 0.5317, 0.562, 0.571, 0.5988, 0.5765, 0.535, 0.5588, 0.588, 0.5695, 0.5725, 0.556, 0.6158, 0.5972, 0.585, 0.566, 0.5955, 0.5618]
"""

# TODO EXP-2 changes:
#  1. Make all augmentations with probability 0.5 (each), no augmentations at all with probability = 0.2+1/16
#  2. Faster-RCNN original #parameters = 41304286 >>
#       New one with Non-FrozenBN2d = 41132062 (requires_grad=False) / 41357406
#       This added 53120 features (BN2d, unfreeze some layers in backbone)
#  3. Random crop with only > 0.5 overlap (before it was accepted also 0., 0.1, 0.3)
#  4. box_detections_per_img=1
#  5. grad clip


# TODO detection mechanism
#  1. set detection threshold on scores before taking most-left
#  2. take 2nd most-left if it's score is much higher than the 1st most-left
#  3. weighted sum (on the scores) for few boxes (e.g. pred_box = 0.6box1 + 0.4box2)

"""
for name, parm in model.named_parameters():
    if parm.requires_grad == False:
        print(name)
"""

if __name__ == '__main__':
    main()
