# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json

import torch
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader
from torch.utils.data.sampler import BatchSampler, Sampler

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import create_transform
from torchvision.datasets.vision import VisionDataset

from PIL import Image

import torch.utils.data as data
import utils
import itertools
import random
from augment import GaussianBlur


from AutoAugment.autoaugment import ImageNetPolicy
from torch.utils.data import ConcatDataset


class MultiImageFolder(data.Dataset):
    def __init__(self, dataset_list, transform, args, loader=default_loader,
                 known_data_source=True, is_test=False) -> None:
        super().__init__()
        self.loader = loader
        self.transform = transform
        self.args = args

        samples_list = [x.samples for x in dataset_list]
        classes_list = [x.classes for x in dataset_list]
        self.classes_list = classes_list
        self.dataset_list = dataset_list
        self.classes = [y for x in self.classes_list for y in x]

        self.gauss = transforms.Compose([transforms.GaussianBlur(kernel_size=3),transforms.ToTensor()])

        start_id = 0
        self.samples = []
        for dataset_id, (samples, classes) in enumerate(zip(samples_list, classes_list)):
            for i, data in enumerate(samples):
                if not is_test:
                    # concat the taxonomy of all datasets
                    if data[1] == -1:
                        img = data[0]
                        self.samples.append((img, None, dataset_id))
                    else:
                        img, target = data[:2]
                        self.samples.append((img, target+start_id, dataset_id))
                        samples[i] = (img, target+start_id)
                else:
                    img = data
                    self.samples.append((img, None, dataset_id))
            start_id += len(classes)

    def __len__(self, ):
        return len(self.samples)

    def __getitem__(self, index):
        """
        Returns:
            sample: the tensor of the input image
            target: a int tensor of class id
            dataset_id: a int number indicating the dataset id
        """
        path, target, dataset_id = self.samples[index]
        while True:
            try:
                sample = self.loader(path)
            except PIL.UnidentifiedImageError:
                image_name = path.split('/')[-1]
                print(f"{image_name} is broke!!!")
                continue
            else:
                break

        # 0 is labeled
        # 1 is unlabeled
        if target != None:

            if self.transform is not None:
                sample = self.transform(sample)

            return sample, target, 0, dataset_id

        else:

            sample_star = self.gauss(sample)

            return sample, sample_star, 1, dataset_id



class Unlabel_MultiImageFolder(data.Dataset):
    def __init__(self, dataset_list, args, loader=default_loader,
                 known_data_source=True, is_test=False) -> None:
        super().__init__()
        self.loader = loader
        #self.transform = transform
        self.args = args

        samples_list = [x.samples for x in dataset_list]
        classes_list = [x.classes for x in dataset_list]
        self.classes_list = classes_list
        self.dataset_list = dataset_list
        self.classes = [y for x in self.classes_list for y in x]

        start_id = 0
        self.samples = []
        for dataset_id, (samples, classes) in enumerate(zip(samples_list, classes_list)):
            for i, data in enumerate(samples):
                if not is_test:
                    # concat the taxonomy of all datasets
                    img, target = data[:2]
                    self.samples.append((img, target+start_id, dataset_id))
                    samples[i] = (img, target+start_id)
                else:
                    img = data
                    self.samples.append((img, None, dataset_id))
            start_id += len(classes)

        self.totensor = transforms.Compose([transforms.ToTensor()])
        self.gauss = transforms.Compose([transforms.GaussianBlur(kernel_size=3),transforms.ToTensor()])

    def __len__(self, ):
        return len(self.samples)

    def __getitem__(self, index):
        """
        Returns:
            sample: the tensor of the input image
            target: a int tensor of class id
            dataset_id: a int number indicating the dataset id
        """
        path, target, dataset_id = self.samples[index]
        while True:
            try:
                sample = self.loader(path)
            except PIL.UnidentifiedImageError:
                image_name = path.split('/')[-1]
                print(f"{image_name} is broke!!!")
                continue
            else:
                break
        

        sample_origin = self.totensor(sample)
        sample_star = self.gauss(sample)

        return sample_origin, sample_star, dataset_id

class TestFolder(data.Dataset):
    def __init__(self, image_root, transform, loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.samples = []
        self.classes = os.listdir(os.path.join(image_root, 'train'))
        image_root = os.path.join(image_root, 'test')

        for file_name in os.listdir(image_root):
            self.samples.append(os.path.join(image_root, file_name))

    def __len__(self, ):
        return len(self.samples)

    def get_image_id(self, path):
        file_name = path.split('/')[-1]
        id_name = file_name.split('.')[0]
        return int(id_name)

    def __getitem__(self, index):
        """
        Returns:
            sample: the tensor of the input image
            image_id: a int number indicating the image id
        """
        path = self.samples[index]
        target = None
        image_id = self.get_image_id(path)
        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, image_id


class UnLablelFolder(data.Dataset):
    def __init__(self, image_root, transform, loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.samples = []
        self.classes = os.listdir(os.path.join(image_root, 'train'))
        self.class_to_idx = {'unlabel':-1}
        self.targets=[]
        image_root = os.path.join(image_root, 'unlabel')

    # self.samples store pathes
        i = 0
        for file_name in os.listdir(image_root):
            self.samples.append((os.path.join(image_root, file_name),-1))
            self.targets.append(-1)
            
            i += 1

    def __len__(self, ):
        return len(self.samples)

    def get_image_id(self, path):
        file_name = path.split('/')[-1]
        id_name = file_name.split('.')[0]
        return int(id_name)

    def __getitem__(self, index):
        """
        Returns:
            sample: the tensor of the input image
            image_id: a int number indicating the image id
        """
        path = self.samples[index]
        target = None
        image_id = self.get_image_id(path)
        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, image_id


def merge_datasets(dataset, sub_dataset):
    '''
        需要合并的Attributes:
            classes (list): List of the class names sorted alphabetically.
            class_to_idx (dict): Dict with items (class_name, class_index).
            samples (list): List of (sample path, class_index) tuples
            targets (list): The class_index value for each image in the dataset
    '''
    #print(dataset.classes)
    # 合并 classes
    dataset.classes.extend(sub_dataset.classes)
    dataset.classes = sorted(list(set(dataset.classes)))
    #print(dataset.classes)
    # 合并 class_to_idx
    dataset.class_to_idx.update(sub_dataset.class_to_idx)
    #print(f"CLASS2IDX TYPE:{dataset.class_to_idx}")
    # 合并 samples
    dataset.samples.extend(sub_dataset.samples)
    # 合并 targets
    dataset.targets.extend(sub_dataset.targets)
    #print(f"TARGET TYPE:{dataset.targets}")
    return dataset


def build_dataset(is_unlabel,is_train, args):
    is_test = not is_train and args.test_only
    if is_unlabel:
        is_test=False
        is_train=False
    transform = build_transform(is_train, args)

    dataset_list = []
    nb_classes = 0

    if is_unlabel:
        for dataset in args.dataset_list:
            root = os.path.join(args.data_path, dataset)
            dataset=UnLablelFolder(root,transform=transform)
            print("Success!")
            dataset_list.append(dataset)
            nb_classes += len(dataset.classes)
        multi_dataset = Unlabel_MultiImageFolder(dataset_list, args, is_test=True)
        return multi_dataset, nb_classes, None

    elif is_test:
        for dataset in args.dataset_list:
            root = os.path.join(args.data_path, dataset)
            dataset = TestFolder(root, transform=transform)
            dataset_list.append(dataset)
            nb_classes += len(dataset.classes)
        multi_dataset = MultiImageFolder(dataset_list, transform, args, is_test=True)
        return multi_dataset, nb_classes, None
    else:
        for dataset in args.dataset_list:
            root = os.path.join(args.data_path, dataset,
                                'train' if is_train else 'val')
            unlabel_root = os.path.join(args.data_path, dataset)
            dataset = datasets.ImageFolder(root, transform=transform)
            #print(len(dataset.classes))
            #print(f"11111111 :: {dataset[0]}")
            original_dataset=ImageFolder(root,transform=transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]))
            #print(len(original_dataset.classes))
            #print(f"33333 :: {original_dataset[0]}")
            
            unlabel = UnLablelFolder(unlabel_root,transform=transforms.Compose([transforms.ToTensor()]))
            dataset=merge_datasets(dataset,original_dataset) 
            dataset=merge_datasets(dataset,unlabel)
            dataset_list.append(dataset)
            #print(len(dataset.classes))    
            #print(f"222222222222 :: {dataset[0]}")       
            nb_classes += len(dataset.classes)
            #print(nb_classes)

        multi_dataset = MultiImageFolder(
            dataset_list, transform, args, known_data_source=args.known_data_source)

        return multi_dataset, nb_classes


def build_transform(is_train, args):
    """ resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=0.3,
            auto_augment='rand-m9-mstd0.5-inc1',
            interpolation='bicubic',
            re_prob=0.25,
            re_mode='pixel',
            re_count=1,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform
    t = []
    if resize_im:
        size = int(args.input_size / args.eval_crop_ratio)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
     
    return transforms.Compose(t) """
    if is_train:
        transform_list= [transforms.RandomResizedCrop(224), 
                         transforms.RandomHorizontalFlip(), 
                         ImageNetPolicy(), 
                         transforms.ToTensor(), 
                         transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)]
    else:
        transform_list = [
            transforms.Resize(args.input_size),
            transforms.CenterCrop(args.input_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ]
    return transforms.Compose(transform_list)


class GroupedDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, batch_sizes, num_datasets):
        """
        Group images from the same dataset into a batch
        """
        self.dataset = dataset
        self.batch_sizes = batch_sizes
        self._buckets = [[] for _ in range(num_datasets)]

    def __len__(self):
        return len(self.dataset) // self.batch_sizes

    def __iter__(self):
        iter_id = 0
        while True:
            for d in self.dataset:
                image, target, dataset_id = d
                bucket = self._buckets[dataset_id]
                bucket.append(d)
                if len(bucket) == self.batch_sizes:
                    images, targets, dataset_ids = list(zip(*bucket))
                    images = torch.stack(images)
                    targets = torch.tensor(targets)
                    dataset_ids = torch.tensor(dataset_ids)
                    del bucket[:]
                    yield images, targets, dataset_ids
                    iter_id += 1
                    if iter_id == len(self):
                        return
