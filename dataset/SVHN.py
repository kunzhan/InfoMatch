import logging
import math
import torch

import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms
import random

from .randaugment_cutmix import RandAugment
from copy import deepcopy
from dataset.mix_1 import rand_bbox

logger = logging.getLogger(__name__)

svhn_mean = (0.4380, 0.4440, 0.4730)
svhn_std = (0.1751, 0.1771, 0.1744)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)


def get_svhn(args, root, svhn_extra=True):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=svhn_mean, std=svhn_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=svhn_mean, std=svhn_std)
    ])

    if svhn_extra:  # train+extra
        
        train_dataset = datasets.SVHN(root, split='train', download=True)#, transform=transforms.ToTensor()) 
        extra_dataset = datasets.SVHN(root, split='extra', download=True)#, transform=transforms.ToTensor())

        combined_data = np.concatenate((train_dataset.data, extra_dataset.data), axis=0)
        combined_labels = np.concatenate((train_dataset.labels, extra_dataset.labels), axis=0)

        train_labeled_idxs, train_unlabeled_idxs = x_u_split(args, combined_labels)

        train_labeled_dataset = SVHNDataset(
            root, combined_data, combined_labels, train_labeled_idxs, split='train',
            transform=transform_labeled)

        train_unlabeled_dataset = SVHNDataset(
            root, combined_data, combined_labels, train_unlabeled_idxs, split='train',
            transform=TransformFixMatch(mean=svhn_mean, std=svhn_std))  # 包含无标签样本 len(set(unlabeled_idx))=73257

    else:  # train_only
        base_dataset = datasets.SVHN(root, split='train', download=True)

        train_labeled_idxs, train_unlabeled_idxs = x_u_split(
            args, base_dataset.labels)

        train_labeled_dataset = SVHNSSL(
            root, train_labeled_idxs, split='train',
            transform=transform_labeled)

        train_unlabeled_dataset = SVHNSSL(
            root, train_unlabeled_idxs, split='train',
            transform=TransformFixMatch(mean=svhn_mean, std=svhn_std))

    test_dataset = datasets.SVHN(
        root, split = 'test', transform=transform_val, download=True)

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

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)

    return labeled_idx, unlabeled_idx


class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect')])
        # self.strong = transforms.Compose([
        #     transforms.RandomHorizontalFlip(),
        #     transforms.RandomCrop(size=32,
        #                           padding=int(32*0.125),
        #                           padding_mode='reflect'),
        #     RandAugment(3, 5)])
        self.strong1 = transforms.Compose([
            RandAugment(3, 5, flag_using_random_num=True)])
        self.strong2 = transforms.Compose([
            RandAugment(3, 5, flag_using_random_num=True)])
            #RandAugmentMC(n=2, m=10)])
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

class SVHNSSL(datasets.SVHN):
    def __init__(self, root, indexs, split='train',
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, split=split,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.labels = np.array(self.labels)[indexs]
        self.data = self.data.transpose([0, 2, 3, 1])

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class SVHNDataset(datasets.SVHN):
    def __init__(self, root, data, labels, indexs, split='train',
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, split=split,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        self.data = data
        self.labels = labels
        if indexs is not None:
            self.data = self.data[indexs]
            self.labels = np.array(self.labels)[indexs]
        self.data = self.data.transpose([0, 2, 3, 1])

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target