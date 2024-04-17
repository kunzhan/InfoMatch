import logging
import math

import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms

from .randaugment_cutmix import RandAugment
from copy import deepcopy
from dataset.mix_1 import rand_bbox

logger = logging.getLogger(__name__)

imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std = (0.229, 0.224, 0.225)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

class ImagenetDataset(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform, ulb, num_labels=-1):
        super().__init__(root, transform)
        self.ulb = ulb
        self.num_labels = num_labels
        is_valid_file = None
        extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
        classes, class_to_idx = self._find_classes(self.root)  # 根据路径：得到，文件名，数字索引。当然它将文件名表示为类别。
        samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.loader = default_loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        if self.ulb:
            self.strong_transform1 = copy.deepcopy(transform)
            self.strong_transform1.transforms.insert(0, RandAugment(3, 5))
            self.strong_transform2 = copy.deepcopy(transform)
            self.strong_transform2.transforms.insert(0, RandAugment(3, 5))

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample_transformed = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return (index, sample_transformed, target) if not self.ulb else (
            index, sample_transformed, self.strong_transform(sample))

    def make_dataset(
            self,
            directory,
            class_to_idx,
            extensions=None,
            is_valid_file=None,
    ):
        instances = []
        directory = os.path.expanduser(directory)
        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none or both_something:
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
        if extensions is not None:
            def is_valid_file(x: str) -> bool:
                return x.lower().endswith(extensions)

        lb_idx = {}

        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                random.shuffle(fnames)
                if self.num_labels != -1:
                    fnames = fnames[:self.num_labels]
                if self.num_labels != -1:
                    lb_idx[target_class] = fnames
                for fname in fnames:
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        item = path, class_index
                        instances.append(item)
        if self.num_labels != -1:
            with open('./sampled_label_idx.json', 'w') as f:
                json.dump(lb_idx, f)
        del lb_idx
        gc.collect()
        return instances


class ImageNetLoader:
    def __init__(self, root_path, num_labels=-1, num_class=1000):
        self.root_path = os.path.join(root_path, 'imagenet')
        self.num_labels = num_labels // num_class  # //：整除

    def get_transform(self, train, ulb):  # ulb: 是否是unlabeld data
        if train:
            transform = transforms.Compose([
                transforms.Resize([256, 256]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(224, padding=4, padding_mode='reflect'),
                transforms.ToTensor(),
                transforms.Normalize(imagenet_mean, imagenet_std)])
        else:
            transform = transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(imagenet_mean, imagenet_std)])
        return transform

    def get_lb_train_data(self):
        transform = self.get_transform(train=True, ulb=False)
        data = ImagenetDataset(root=os.path.join(self.root_path, "train"), transform=transform, ulb=False,
                               num_labels=self.num_labels)
        return data

    def get_ulb_train_data(self):
        transform = self.get_transform(train=True, ulb=True)
        data = ImagenetDataset(root=os.path.join(self.root_path, "train"), transform=transform, ulb=True)
        return data

    def get_lb_test_data(self):
        transform = self.get_transform(train=False, ulb=False)
        data = ImagenetDataset(root=os.path.join(self.root_path, "val"), transform=transform, ulb=False)
        return data