from dataset import *
from model import get_fasterrcnn_resnet50_fpn
import constants
import warnings
# For drawing onto the image
import numpy as np
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
import torchvision.transforms as T
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=DeprecationWarning)


def display_image(image, title=''):
    fig = plt.figure(figsize=(10, 10), dpi=200)
    plt.grid(False)
    plt.title(title, fontdict={'fontsize': 14})
    plt.imshow(image)
    # plt.savefig(f'{title}.png')
    plt.show()


def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color,
                               font,
                               thickness=1,
                               display_str_list=()):
    """Adds a bounding box to an image."""
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
    draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
               (left, top)],
              width=thickness,
              fill=color)

    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = top + total_display_str_height
    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                        (left + text_width, text_bottom)],
                       fill=color)
        draw.text((left + margin, text_bottom - text_height - margin),
                  display_str,
                  fill="black",
                  font=font)
        text_bottom -= text_height - 2 * margin


def draw_boxes(image, boxes, class_names, scores, max_boxes=10, min_score=0.1):
    """Overlay labeled boxes on an image with formatted scores and label names."""
    colors = list(ImageColor.colormap.values())
    colors = {'uncovered': '#ff7f7f', 'covered': '#7fffd4', 'other': '#f07fff'}
    try:
        font = ImageFont.truetype("LiberationSansNarrow-Bold.ttf", 12)
    except IOError:
        print("Font not found, using default font.")
        font = ImageFont.load_default()

    for i in range(min(boxes.shape[0], max_boxes)):
        if scores[i] >= min_score:
            xmin, ymin, xmax, ymax = tuple(boxes[i])
            display_str = "{}: {}%".format(
                class_names[i].decode("ascii") if type(class_names[i]) == bytes else class_names[i],
                int(100 * scores[i]))
            color = colors[class_names[i]]
            image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
            draw_bounding_box_on_image(
                image_pil,
                ymin,
                xmin,
                ymax,
                xmax,
                color,
                font,
                display_str_list=[display_str])
            np.copyto(dst=image, src=np.array(image_pil))
    return image


if __name__ == '__main__':
    model = get_fasterrcnn_resnet50_fpn(weights_path='checkpoint_fasterrcnn_epoch=48.pth.tar')
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # test
    dataset = TrucksDataset(data_folder=constants.TEST_DIRECTORY_PATH, split='test')
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=20, shuffle=False, collate_fn=collate_fn)

    transform = T.ToPILImage()
    images, targets = next(iter(test_loader))
    images = [image.to(device) for image in images]

    res = model(images)
    voc_labels = ('uncovered', 'covered', 'other')
    label_map = {v + 1: k for v, k in enumerate(voc_labels)}

    for k in range(len(images)):
        img = np.array(transform(images[k]))
        display_image(draw_boxes(image=img,
                                 boxes=res[k].get('boxes').cpu().detach().numpy() / 224,
                                 class_names=[label_map[label] for label in res[k].get('labels').cpu().numpy()],
                                 scores=res[k].get('scores').cpu().detach().numpy()
                                 ),
                      f"image_id = {targets[k]['image_id'].item()}"
                      )
