import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models import resnet
from torchvision.ops import misc as misc_nn_ops
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection.faster_rcnn import FasterRCNN


# TODO
#   Use fasterrcnn_resnet50_fpn_v2
#   Maybe consider https://rwightman.github.io/pytorch-image-models/models/inception-resnet-v2/

def resnet_fpn_backbone(backbone_name, pretrained):
    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained,
        norm_layer=misc_nn_ops.BatchNorm2d)

    return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}

    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [
        in_channels_stage2,
        in_channels_stage2 * 2,
        in_channels_stage2 * 4,
        in_channels_stage2 * 8,
    ]
    out_channels = 256
    return BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels)


def fasterrcnn_resnet50_fpn(num_classes=91, pretrained_backbone=False, **kwargs):
    backbone = resnet_fpn_backbone('resnet50', pretrained_backbone)
    model = FasterRCNN(backbone, num_classes, **kwargs)
    return model


def get_fasterrcnn_resnet50_fpn(weights_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TrucksDataset mean and std
    mean = [0.5244, 0.4904, 0.4781]
    std = [0.2642, 0.2608, 0.2561]

    # Initialize model
    model = fasterrcnn_resnet50_fpn(pretrained_backbone=False,
                                    image_mean=mean,
                                    image_std=std,
                                    min_size=224,
                                    max_size=224,
                                    box_detections_per_img=1)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=3)  # classes + 1 (background)

    if weights_path:
        model.load_state_dict(torch.load(weights_path)['state_dict'])

    return model.to(device)


if __name__ == '__main__':
    # test this module
    model = get_fasterrcnn_resnet50_fpn()
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
