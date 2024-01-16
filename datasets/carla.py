import os
from pathlib import Path
from PIL import Image
import torch
import torch.utils.data
from torchvision import transforms
from torchvision.datasets.vision import VisionDataset
import datasets.transforms as T


class Carla_Dataset(VisionDataset):
    def __init__(self, root, transforms):
        super(Carla_Dataset, self).__init__(root, transforms)
        self.img_names = os.listdir(root)
        self._transforms = transforms

    def __getitem__(self, index):
        # regiter images in a list
        # read_img_folder, pil.imread("xxx") -> list
        # img_list = []

        # read json folder, import json, json.read("xxx"), write a pseudo function to return fake anno
        # anno_lits = get_anno()
        img_path = os.path.join(self.root, self.img_names[index])
        img = Image.open(img_path).convert("RGB")
        tgt = self.get_anno()
        if self._transforms is not None:
            img, tgt = self._transforms(img, tgt)
        return img, tgt

    def __len__(self):
        return len(self.img_names)

    def get_anno(self, anno_file=None):
        targets = {
            "boxes": torch.tensor(
                [[0.5325, 0.9101, 0.2119, 0.1546], [0.2679, 0.9204, 0.2207, 0.1592]]
            ),
            "labels": torch.tensor([15, 15]),
            "image_id": torch.tensor([86835]),
            "area": torch.tensor([8161.0596, 8446.8818]),
            "iscrowd": torch.tensor([0, 0]),
            "orig_size": torch.tensor([720, 1280]),
            "size": torch.tensor([512, 767]),
        }
        return targets


def make_carla_transforms():
    normalize = T.Compose(
        [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )

    return T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            normalize,
        ]
    )


def build_carla(image_set, args):
    root_path = Path(args.carla_path)
    assert root_path.exists(), f"provided carla path {root_path} does not exist"
    PATHS = {"test": root_path}
    img_folder = PATHS[image_set]
    dataset = Carla_Dataset(root=img_folder, transforms=make_carla_transforms())
    return dataset
