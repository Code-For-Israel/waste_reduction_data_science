import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models import resnet
from torchvision.ops import misc as misc_nn_ops
from torchvision.models.detection.backbone_utils import BackboneWithFPN, resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FasterRCNN
from constants import TRUCKS_DATASET_MEAN, TRUCKS_DATASET_STD


# TODO
#   Use fasterrcnn_resnet50_fpn_v2 from: https://github.com/pytorch/vision/blob/main/torchvision/models/detection/faster_rcnn.py
#   https://github.com/pytorch/vision/blob/main/torchvision/models/detection/backbone_utils.py
#   WideResNet? "wide_resnet50_2" as backbone_name
#   Or maybe consider https://rwightman.github.io/pytorch-image-models/models/inception-resnet-v2/

# TODO Maybe take the original function from torchvision?
# def resnet_fpn_backbone(backbone_name, pretrained):
#     backbone = resnet.__dict__[backbone_name](
#         pretrained=pretrained,
#         norm_layer=misc_nn_ops.FrozenBatchNorm2d)  # TODO Try without FrozenBatchNorm2d >> BatchNorm2d
#
#     return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
#
#     in_channels_stage2 = backbone.inplanes // 8
#     in_channels_list = [
#         in_channels_stage2,
#         in_channels_stage2 * 2,
#         in_channels_stage2 * 4,
#         in_channels_stage2 * 8,
#     ]
#     out_channels = 256
#     return BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels)


def fasterrcnn_resnet50_fpn(num_classes=4, pretrained_backbone=True, **kwargs):
    backbone = resnet_fpn_backbone('resnet50', pretrained_backbone)
    model = FasterRCNN(backbone, num_classes, **kwargs)
    return model


def get_fasterrcnn_resnet50_fpn(weights_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TrucksDataset mean and std
    # TODO change in every place, maybe set as constants and find how to calculate after we have all the dataset saved
    mean = TRUCKS_DATASET_MEAN
    std = TRUCKS_DATASET_STD

    # Initialize model
    model = fasterrcnn_resnet50_fpn(pretrained_backbone=True,  # TODO What we want?
                                    image_mean=mean,
                                    image_std=std,
                                    min_size=224,
                                    max_size=224,
                                    # TODO more parameters to adjust? overlap?
                                    box_score_thresh=0.5,
                                    # 0.5 is default (during inference,
                                    # only return proposals with a classification score greater than box_score_thresh)
                                    box_detections_per_img=100,  # 100 is the default
                                    # (maximum number of detections per image, for all classes.)
                                    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=4)  # classes + 1 (background)

    if weights_path:
        model.load_state_dict(torch.load(weights_path)['state_dict'])

    return model.to(device)


if __name__ == '__main__':
    # test this module
    # create model
    model = get_fasterrcnn_resnet50_fpn()
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    # load model from checkpoint
    model = get_fasterrcnn_resnet50_fpn(weights_path='checkpoint_fasterrcnn_epoch=48.pth.tar')
