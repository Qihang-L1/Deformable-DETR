import os
from pathlib import Path
from PIL import Image
import torch
import torch.utils.data
from torchvision.datasets.vision import VisionDataset
import datasets.transforms as transforms



class Carla_Dataset(VisionDataset):
    """
    A dataset class for the Carla Dataset.

    Args:
        root (string): Root directory of the Carla Dataset.
        transforms (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """

    def __init__(self, root, transforms):
        super(Carla_Dataset, self).__init__(root, transforms)
        self.img_names = os.listdir(root)
        self._transforms = transforms

    def __getitem__(self, index):
        """
        Retrieves the image and target annotation at the given index.

        Args:
            index (int): Index of the image and target annotation to retrieve.

        Returns:
            tuple: A tuple containing the image and target annotation.
        """
        img_path = os.path.join(self.root, self.img_names[index])
        img = Image.open(img_path).convert("RGB")
        tgt = self.get_anno()
        if self._transforms is not None:
            img, tgt = self._transforms(img, tgt)
        return img, tgt, img_path

    def __len__(self):
        """
        Returns the total number of images in the dataset.

        Returns:
            int: The total number of images in the dataset.
        """
        return len(self.img_names)

    def get_anno(self):
        """
        Retrieves the target annotation for an image.

        Returns:
            dict: A dictionary containing the target annotation, which is now a dummy.
            The key value "orig_size" representing original size of the image must be given correctly.
        """
        targets = {
            "orig_size": torch.tensor([720, 1280]),
        }
        return targets


def make_carla_transforms():
    normalize = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    """ 
    The image transformation and normalization used in test dataset are the same as the ones used in evaluation dataset.
    """
    return transforms.Compose(
        [
            transforms.RandomResize([720], max_size=1333),
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
