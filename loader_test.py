import os

import torch
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader
from torch.utils.data.sampler import BatchSampler, Sampler


x = default_loader('/remote-home/share/course23/aicourse_dataset_final/10shot_cifar100_20200721/unlabel/1001.jpg')
t_list = [transforms.ToTensor()]
trans = transforms.Compose(t_list)
x = trans(x)

print(x)