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

def get_imagenet(args, root):
    transform_labeled = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=224,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    transform_val = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    print(root)

    # base_dataset = datasets.ImageNet(root, split= 'train', download=False)
    base_dataset = datasets.ImageFolder(root+'/train')

    # import ipdb
    # ipdb.set_trace()
    # base_dataset = datasets.ImageFolder(root+'/Data/CLS-LOC/train')
    # datasets.CIFAR10(root, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets)

    train_labeled_dataset = ImageNetSSL(
        root+'/train', train_labeled_idxs, 
        transform=transform_labeled)

    # import ipdb
    # ipdb.set_trace()

    train_unlabeled_dataset = ImageNetSSL(
        root+'/train', train_unlabeled_idxs,
        transform=TransformFixMatch(mean=imagenet_mean, std=imagenet_std))

    test_dataset = datasets.ImageFolder(
        root+'/val', transform=transform_val)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset

def x_u_split(args, labels):
    # 将数据集中的样本划分为有标签数据和无标签数据的索引
    label_per_class = args.num_labeled // args.num_classes
    labels = np.array(labels)
    labeled_idx = []
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == args.num_labeled

    if args.expand_labels or args.num_labeled < args.batch_size * args.world_size:
        num_expand_x = math.ceil(
            (args.batch_size * args.world_size) * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    # print(len(labeled_idx), len(unlabeled_idx))
    return labeled_idx, unlabeled_idx


class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.Resize([256, 256]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=224,
                              padding=int(32*0.125),
                              padding_mode='reflect')])
        self.strong1 = transforms.Compose([
            RandAugment(3, 5, flag_using_random_num=True)])
        self.strong2 = transforms.Compose([
            RandAugment(3, 5, flag_using_random_num=True)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak, strong1, strong2 = deepcopy(x), deepcopy(x), deepcopy(x)
        weak = self.weak(x)
        strong1 = self.strong1(weak)
        strong2 = self.strong2(weak)
        weak = self.normalize(weak)
        bbox_w, lam_w = rand_bbox(weak.size())
        strong1 = self.normalize(strong1)
        strong2 = self.normalize(strong2)
        return weak, strong1, strong2, bbox_w, lam_w

class ImageNetSSL(datasets.ImageFolder):
    def __init__(self, root, indexs,
                 transform=None, target_transform=None):
        super().__init__(root, 
                         transform=transform,
                         target_transform=target_transform)
        if indexs is not None:
            # self.samples = self.samples[indexs]
            self.samples = [self.samples[indexs[i]] for i in range(len(indexs))]
            # self.samples = np.array(self.samples)[indexs].tolist()
            # self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.loader(path)
        # img, target = self.samples[index], self.targets[index]
        # img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        
        # import ipdb
        # ipdb.set_trace()

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, np.array(target)

# class ImageNetSSL(datasets.ImageNet):
#     def __init__(self, root, indexs, split='train',
#                  transform=None, target_transform=None,
#                  download=False):
#         super().__init__(root, split=split,
#                          transform=transform,
#                          target_transform=target_transform,
#                          download=download)
#         if indexs is not None:
#             self.data = self.data[indexs]
#             self.targets = np.array(self.targets)[indexs]

#     def __getitem__(self, index):
#         img, target = self.data[index], self.targets[index]
#         img = Image.fromarray(img)

#         if self.transform is not None:
#             img = self.transform(img)

#         if self.target_transform is not None:
#             target = self.target_transform(target)

#         return img, target
