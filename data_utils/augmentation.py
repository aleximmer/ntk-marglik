import torch
from torchvision import transforms

cifar_mean = (0.49139968, 0.48215841, 0.44653091)
cifar_std = (0.24703223, 0.24348513, 0.26158784)
imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std = (0.229, 0.224, 0.225)

CIFAR_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ]
)

MNIST_transform = transforms.ToTensor()

ImageNet_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std)
    ]
)
