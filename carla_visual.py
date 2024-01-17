"""
visualization of the inference results, pending for further evaluation.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


def carla_vis(dic, lbl, pth):
    os.makedirs("output_images", exist_ok=True)
    image = Image.open(pth[0])
    filename = os.path.basename(pth[0])
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    for box, label in zip(dic["boxes"].cpu(), lbl.cpu()):
        x0, y0, x1, y1 = box[0], box[1], box[2], box[3]
        cat = Label_map(label.item())
        rect = patches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0, linewidth=2, edgecolor="r", facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(x0, y0, f"{cat}", color="r")
    plt.savefig(f"output_images/{filename}")
    plt.close()


def Label_map(number):
    label_map = {
        1: "person",
        2: "bicycle",
        3: "car",
        4: "motorbike",
        5: "aeroplane",
        6: "bus",
        7: "train",
        8: "truck",
        9: "boat",
        10: "traffic light",
        11: "fire hydrant",
        13: "stop sign",
        14: "parking meter",
        15: "bench",
        16: "bird",
        17: "cat",
        18: "dog",
        19: "horse",
        20: "sheep",
        21: "cow",
        22: "elephant",
        23: "bear",
        24: "zebra",
        25: "giraffe",
        27: "backpack",
        28: "umbrella",
        31: "handbag",
        32: "tie",
        33: "suitcase",
        34: "frisbee",
        35: "skis",
        36: "snowboard",
        37: "sports ball",
        38: "kite",
        39: "baseball bat",
        40: "baseball glove",
        41: "skateboard",
        42: "surfboard",
        43: "tennis racket",
        44: "bottle",
        46: "wine glass",
        47: "cup",
        48: "fork",
        49: "knife",
        50: "spoon",
        51: "bowl",
        52: "banana",
        53: "apple",
        54: "sandwich",
        55: "orange",
        56: "broccoli",
        57: "carrot",
        58: "hot dog",
        59: "pizza",
        60: "donut",
        61: "cake",
        62: "chair",
        63: "sofa",
        64: "pottedplant",
        65: "bed",
        67: "diningtable",
        70: "toilet",
        72: "tvmonitor",
        73: "laptop",
        74: "mouse",
        75: "remote",
        76: "keyboard",
        77: "cell phone",
        78: "microwave",
        79: "oven",
        80: "toaster",
        81: "sink",
        82: "refrigerator",
        84: "book",
        85: "clock",
        86: "vase",
        87: "scissors",
        88: "teddy bear",
        89: "hair drier",
        90: "toothbrush",
    }
    return label_map.get(number, "unkonown")
