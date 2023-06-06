import numpy as np
import torch
from torchvision.datasets import ImageFolder
from typing import Union, Callable 
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets

LOG2 = np.log(2.0)


class RotatedMNIST(datasets.MNIST):
    """ Rotated MNIST class.
        Wraps regular pytorch MNIST and rotates each image randomly between given angle using fixed seed. """

    def __init__(self, root: str, degree: float, mode: str = 'bilinear',
                 train: bool = True, transform: Union[Callable, type(None)] = None,
                 target_transform: Union[Callable, type(None)] = None, download: bool = False):
        super().__init__(root, train, transform, target_transform, download)
        """
        Args:
            root (string): Root directory of dataset where ``MNIST/processed/training.pt``
                and  ``MNIST/processed/test.pt`` exist.
            degree (float): Amount of rotation in degrees
            mode (float): Mode used for interpolation (nearest|bilinear|bicubic)
            train (bool, optional): If True, creates dataset from ``training.pt``,
                otherwise from ``test.pt``.
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
                same arguments as torchvision.see arguments of torchvision.dataset.MNIST)
        """
        # Use fixed seed (0 for test set, 1 for training set)
        torch.manual_seed(int(train))

        # Sample radians
        rad = np.radians(degree)
        thetas = (torch.rand(len(self.data)) * 2 - 1) * rad

        # Create rotation matrices
        c, s = torch.cos(thetas), torch.sin(thetas)
        rot_matrices = torch.stack((torch.stack((c, -s, c*0), 0),
                                    torch.stack((s, c, s*0), 0)), 0).permute(2, 0, 1)

        # Create resampling grid
        rot_grids = F.affine_grid(rot_matrices, self.data.unsqueeze(1).shape, align_corners=False)

        # Rotate images using 'mode' interpolation (e.g. bilinear sampling)
        xmin, xmax = self.data.min(), self.data.max()
        data_dtype = self.data.dtype

        self.data = F.grid_sample(self.data.unsqueeze(1).float(), rot_grids, align_corners=False, mode=mode)
        self.data = self.data[:, 0].type(data_dtype).clamp(xmin, xmax)


class RotatedFashionMNIST(datasets.FashionMNIST):
    """ Rotated Fashion MNIST dataset.
        Wraps regular pytorch FashionMNIST and rotates each image randomly between given angle using fixed seed. """

    def __init__(self, root: str, degree: float, mode: str = 'bilinear',
                 train: bool = True, transform: Union[Callable, type(None)] = None,
                 target_transform: Union[Callable, type(None)] = None, download: bool = False):
        super().__init__(root, train, transform, target_transform, download)
        """
        Args:
            root (string): Root directory of dataset where ``MNIST/processed/training.pt``
                and  ``MNIST/processed/test.pt`` exist.
            degree (float): Amount of rotation in degrees
            mode (float): Mode used for interpolation (nearest|bilinear|bicubic)
            train (bool, optional): If True, creates dataset from ``training.pt``,
                otherwise from ``test.pt``.
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
                same arguments as torchvision.see arguments of torchvision.dataset.MNIST)
        """

        # Use fixed seed (0 for test set, 1 for training set)
        torch.manual_seed(int(train))

        # Sample radians
        rad = np.radians(degree)
        thetas = (torch.rand(len(self.data)) * 2 - 1) * rad

        # Create rotation matrices
        c, s = torch.cos(thetas), torch.sin(thetas)
        rot_matrices = torch.stack((torch.stack((c, -s, c*0), 0),
                                    torch.stack((s, c, s*0), 0)), 0).permute(2, 0, 1)

        # Create resampling grid
        rot_grids = F.affine_grid(rot_matrices, self.data.unsqueeze(1).shape, align_corners=False)

        # Rotate images using 'mode' interpolation (e.g. bilinear sampling)
        xmin, xmax = self.data.min(), self.data.max()
        data_dtype = self.data.dtype

        self.data = F.grid_sample(self.data.unsqueeze(1).float(), rot_grids, align_corners=False, mode=mode)
        self.data = self.data[:, 0].type(data_dtype).clamp(xmin, xmax)


class TranslatedMNIST(datasets.MNIST):
    """ MNIST translated by fixed amount using random seed """

    def __init__(self, root: str, amount: float = 8.0, mode: str = 'bilinear',
                 train: bool = True, transform: Union[Callable, type(None)] = None,
                 target_transform: Union[Callable, type(None)] = None, download: bool = False):
        super().__init__(root, train, transform, target_transform, download)
        """
        Args:
            root (string): Root directory of dataset where ``MNIST/processed/training.pt``
                and  ``MNIST/processed/test.pt`` exist.
            amount (float): Amount of translation in pixels
            mode (float): Mode used for interpolation (nearest|bilinear|bicubic)
            train (bool, optional): If True, creates dataset from ``training.pt``,
                otherwise from ``test.pt``.
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
                same arguments as torchvision.see arguments of torchvision.dataset.MNIST)
        """

        torch.manual_seed(int(train))

        rad = np.radians(amount)

        r1 = (torch.rand(len(self.data)) * 2 - 1) * amount / 14.0
        r2 = (torch.rand(len(self.data)) * 2 - 1) * amount / 14.0

        zero = r1 * 0.0
        one = zero + 1.0

        matrices = torch.stack((torch.stack((one, zero, r1), 0),
                                torch.stack((zero, one, r2), 0)), 0).permute(2, 0, 1)

        grids = F.affine_grid(matrices, self.data.unsqueeze(1).shape, align_corners=True)

        xmin, xmax = self.data.min(), self.data.max()
        data_dtype = self.data.dtype

        self.data = F.grid_sample(self.data.unsqueeze(1).float(), grids, align_corners=True, mode=mode)
        self.data = self.data[:, 0].type(data_dtype).clamp(xmin, xmax)


class ScaledMNIST(datasets.MNIST):
    """ MNIST scaled by fixed amount using random seed """

    def __init__(self, root: str, amount: float = LOG2, mode: str = 'bilinear',
                 train: bool = True, transform: Union[Callable, type(None)] = None,
                 target_transform: Union[Callable, type(None)] = None, download: bool = False):
        super().__init__(root, train, transform, target_transform, download)
        """
        Args:
            root (string): Root directory of dataset where ``MNIST/processed/training.pt``
                and  ``MNIST/processed/test.pt`` exist.
            amount (float): Amount of translation in pixels
            mode (float): Mode used for interpolation (nearest|bilinear|bicubic)
            train (bool, optional): If True, creates dataset from ``training.pt``,
                otherwise from ``test.pt``.
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
                same arguments as torchvision.see arguments of torchvision.dataset.MNIST)
        """
        torch.manual_seed(int(train))
        s1 = (torch.rand(len(self.data)) * 2 - 1) * amount

        zero = s1 * 0.0

        matrices = torch.stack((torch.stack((torch.exp(s1), zero, zero), 0),
                                torch.stack((zero, torch.exp(s1), zero), 0)), 0).permute(2, 0, 1)

        grids = F.affine_grid(matrices, self.data.unsqueeze(1).shape, align_corners=True)

        xmin, xmax = self.data.min(), self.data.max()
        data_dtype = self.data.dtype

        self.data = F.grid_sample(self.data.unsqueeze(1).float(), grids, align_corners=True, mode=mode)
        self.data = self.data[:, 0].type(data_dtype).clamp(xmin, xmax)


class RotatedFashionMNIST(datasets.FashionMNIST):
    """ Rotated FashionMNIST class.
        Wraps regular pytorch FashionMNIST and rotates each image randomly between given angle using fixed seed. """

    def __init__(self, root: str, degree: float, mode: str = 'bilinear',
                 train: bool = True, transform: Union[Callable, type(None)] = None,
                 target_transform: Union[Callable, type(None)] = None, download: bool = False):
        super().__init__(root, train, transform, target_transform, download)
        """
        Args:
            root (string): Root directory of dataset where ``FashionMNIST/processed/training.pt``
                and  ``FashionMNIST/processed/test.pt`` exist.
            degree (float): Amount of rotation in degrees
            mode (float): Mode used for interpolation (nearest|bilinear|bicubic)
            train (bool, optional): If True, creates dataset from ``training.pt``,
                otherwise from ``test.pt``.
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
                same arguments as torchvision.see arguments of torchvision.dataset.FashionMNIST)
        """
        # Use fixed seed (0 for test set, 1 for training set)
        torch.manual_seed(int(train))

        # Sample radians
        rad = np.radians(degree)
        thetas = (torch.rand(len(self.data)) * 2 - 1) * rad

        # Create rotation matrices
        c, s = torch.cos(thetas), torch.sin(thetas)
        rot_matrices = torch.stack((torch.stack((c, -s, c*0), 0),
                                    torch.stack((s, c, s*0), 0)), 0).permute(2, 0, 1)

        # Create resampling grid
        rot_grids = F.affine_grid(rot_matrices, self.data.unsqueeze(1).shape, align_corners=False)

        # Rotate images using 'mode' interpolation (e.g. bilinear sampling)
        xmin, xmax = self.data.min(), self.data.max()
        data_dtype = self.data.dtype

        self.data = F.grid_sample(self.data.unsqueeze(1).float(), rot_grids, align_corners=False, mode=mode)
        self.data = self.data[:, 0].type(data_dtype).clamp(xmin, xmax)


class RotatedFashionMNIST(datasets.FashionMNIST):
    """ Rotated Fashion MNIST dataset.
        Wraps regular pytorch FashionMNIST and rotates each image randomly between given angle using fixed seed. """

    def __init__(self, root: str, degree: float, mode: str = 'bilinear',
                 train: bool = True, transform: Union[Callable, type(None)] = None,
                 target_transform: Union[Callable, type(None)] = None, download: bool = False):
        super().__init__(root, train, transform, target_transform, download)
        """
        Args:
            root (string): Root directory of dataset where ``FashionMNIST/processed/training.pt``
                and  ``FashionMNIST/processed/test.pt`` exist.
            degree (float): Amount of rotation in degrees
            mode (float): Mode used for interpolation (nearest|bilinear|bicubic)
            train (bool, optional): If True, creates dataset from ``training.pt``,
                otherwise from ``test.pt``.
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
                same arguments as torchvision.see arguments of torchvision.dataset.FashionMNIST)
        """
        # Use fixed seed (0 for test set, 1 for training set)
        torch.manual_seed(int(train))

        # Sample radians
        rad = np.radians(degree)
        thetas = (torch.rand(len(self.data)) * 2 - 1) * rad

        # Create rotation matrices
        c, s = torch.cos(thetas), torch.sin(thetas)
        rot_matrices = torch.stack((torch.stack((c, -s, c*0), 0),
                                    torch.stack((s, c, s*0), 0)), 0).permute(2, 0, 1)

        # Create resampling grid
        rot_grids = F.affine_grid(rot_matrices, self.data.unsqueeze(1).shape, align_corners=False)

        # Rotate images using 'mode' interpolation (e.g. bilinear sampling)
        xmin, xmax = self.data.min(), self.data.max()
        data_dtype = self.data.dtype

        self.data = F.grid_sample(self.data.unsqueeze(1).float(), rot_grids, align_corners=False, mode=mode)
        self.data = self.data[:, 0].type(data_dtype).clamp(xmin, xmax)


class TranslatedFashionMNIST(datasets.FashionMNIST):
    """ FashionMNIST translated by fixed amount using random seed """

    def __init__(self, root: str, amount: float = 8.0, mode: str = 'bilinear',
                 train: bool = True, transform: Union[Callable, type(None)] = None,
                 target_transform: Union[Callable, type(None)] = None, download: bool = False):
        super().__init__(root, train, transform, target_transform, download)
        """
        Args:
            root (string): Root directory of dataset where ``FashionMNIST/processed/training.pt``
                and  ``FashionMNIST/processed/test.pt`` exist.
            amount (float): Amount of translation in pixels
            mode (float): Mode used for interpolation (nearest|bilinear|bicubic)
            train (bool, optional): If True, creates dataset from ``training.pt``,
                otherwise from ``test.pt``.
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
                same arguments as torchvision.see arguments of torchvision.dataset.FashionMNIST)
        """
        torch.manual_seed(int(train))
        r1 = (torch.rand(len(self.data)) * 2 - 1) * amount / 14.0
        r2 = (torch.rand(len(self.data)) * 2 - 1) * amount / 14.0

        zero = r1 * 0.0
        one = zero + 1.0

        matrices = torch.stack((torch.stack((one, zero, r1), 0),
                                torch.stack((zero, one, r2), 0)), 0).permute(2, 0, 1)

        grids = F.affine_grid(matrices, self.data.unsqueeze(1).shape, align_corners=True)

        xmin, xmax = self.data.min(), self.data.max()
        data_dtype = self.data.dtype

        self.data = F.grid_sample(self.data.unsqueeze(1).float(), grids, align_corners=True, mode=mode)
        self.data = self.data[:, 0].type(data_dtype).clamp(xmin, xmax)


class ScaledFashionMNIST(datasets.FashionMNIST):
    """ FashionMNIST scaled by fixed amount using random seed """

    def __init__(self, root: str, amount: float = LOG2, mode: str = 'bilinear',
                 train: bool = True, transform: Union[Callable, type(None)] = None,
                 target_transform: Union[Callable, type(None)] = None, download: bool = False):
        super().__init__(root, train, transform, target_transform, download)
        """
        Args:
            root (string): Root directory of dataset where ``FashionMNIST/processed/training.pt``
                and  ``FashionMNIST/processed/test.pt`` exist.
            amount (float): Amount of translation in pixels
            mode (float): Mode used for interpolation (nearest|bilinear|bicubic)
            train (bool, optional): If True, creates dataset from ``training.pt``,
                otherwise from ``test.pt``.
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
                same arguments as torchvision.see arguments of torchvision.dataset.FashionMNIST)
        """

        torch.manual_seed(int(train))

        s1 = (torch.rand(len(self.data)) * 2 - 1) * amount
        zero = s1 * 0.0
        matrices = torch.stack((torch.stack((torch.exp(s1), zero, zero), 0),
                                torch.stack((zero, torch.exp(s1), zero), 0)), 0).permute(2, 0, 1)

        grids = F.affine_grid(matrices, self.data.unsqueeze(1).shape, align_corners=True)

        xmin, xmax = self.data.min(), self.data.max()
        data_dtype = self.data.dtype

        self.data = F.grid_sample(self.data.unsqueeze(1).float(), grids, align_corners=True, mode=mode)
        self.data = self.data[:, 0].type(data_dtype).clamp(xmin, xmax)


class RotatedCIFAR10(datasets.CIFAR10):
    """ CIFAR10 rotated by fixed amount using random seed """

    def __init__(self, root: str, degree: float, mode: str = 'bilinear',
                 train: bool = True, transform: Union[Callable, type(None)] = None,
                 target_transform: Union[Callable, type(None)] = None, download: bool = False):
        super().__init__(root, train, transform, target_transform, download)
        """
        Args:
            root (string): Root directory of dataset where ``CIFAR10/processed/training.pt``
                and  ``CIFAR10/processed/test.pt`` exist.
            degree (float): Amount of rotation in degrees
            mode (float): Mode used for interpolation (nearest|bilinear|bicubic)
            train (bool, optional): If True, creates dataset from ``training.pt``,
                otherwise from ``test.pt``.
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
                same arguments as torchvision.see arguments of torchvision.dataset.CIFAR10)
        """

        torch.manual_seed(int(train))

        rad = np.radians(degree)

        thetas = (torch.rand(len(self.data)) * 2 - 1) * rad

        c, s = torch.cos(thetas), torch.sin(thetas)

        rot_matrices = torch.stack((torch.stack((c, -s, c*0), 0),
                                    torch.stack((s, c, s*0), 0)), 0).permute(2, 0, 1)

        xmin, xmax = self.data.min(), self.data.max()
        data_dtype = self.data.dtype

        self.data = torch.Tensor(self.data.transpose(0, 3, 1, 2))
        rot_grids = F.affine_grid(rot_matrices, self.data.shape, align_corners=False)

        self.data = F.grid_sample(self.data.float(), rot_grids, align_corners=False, mode=mode)
        self.data = self.data.clamp(xmin, xmax).numpy().astype(data_dtype)
        self.data = self.data.transpose(0, 2, 3, 1)


class TranslatedCIFAR10(datasets.CIFAR10):
    """ CIFAR10 translated by fixed amount using random seed """

    def __init__(self, root: str, amount: float = 8.0, mode: str = 'bilinear',
                 train: bool = True, transform: Union[Callable, type(None)] = None,
                 target_transform: Union[Callable, type(None)] = None, download: bool = False):
        super().__init__(root, train, transform, target_transform, download)
        """
        Args:
            root (string): Root directory of dataset where ``CIFAR10/processed/training.pt``
                and  ``CIFAR10/processed/test.pt`` exist.
            amount (float): Amount of translation in pixels
            mode (float): Mode used for interpolation (nearest|bilinear|bicubic)
            train (bool, optional): If True, creates dataset from ``training.pt``,
                otherwise from ``test.pt``.
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
                same arguments as torchvision.see arguments of torchvision.dataset.CIFAR10)
        """

        torch.manual_seed(int(train))

        rad = np.radians(amount)

        r1 = (torch.rand(len(self.data)) * 2 - 1) * amount / 14.0
        r2 = (torch.rand(len(self.data)) * 2 - 1) * amount / 14.0

        zero = r1 * 0.0
        one = zero + 1.0

        matrices = torch.stack((torch.stack((one, zero, r1), 0),
                                torch.stack((zero, one, r2), 0)), 0).permute(2, 0, 1)

        xmin, xmax = self.data.min(), self.data.max()
        data_dtype = self.data.dtype

        self.data = torch.Tensor(self.data.transpose(0, 3, 1, 2))
        grids = F.affine_grid(matrices, self.data.shape, align_corners=False)

        self.data = F.grid_sample(self.data.float(), grids, align_corners=False, mode=mode)
        self.data = self.data.clamp(xmin, xmax).numpy().astype(data_dtype)
        self.data = self.data.transpose(0, 2, 3, 1)


class ScaledCIFAR10(datasets.CIFAR10):
    """ CIFAR10 scaled by fixed amount using random seed """

    def __init__(self, root: str, amount: float = LOG2, mode: str = 'bilinear',
                 train: bool = True, transform: Union[Callable, type(None)] = None,
                 target_transform: Union[Callable, type(None)] = None, download: bool = False):
        super().__init__(root, train, transform, target_transform, download)
        """
        Args:
            root (string): Root directory of dataset where ``CIFAR10/processed/training.pt``
                and  ``CIFAR10/processed/test.pt`` exist.
            amount (float): Amount of translation in pixels
            mode (float): Mode used for interpolation (nearest|bilinear|bicubic)
            train (bool, optional): If True, creates dataset from ``training.pt``,
                otherwise from ``test.pt``.
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
                same arguments as torchvision.see arguments of torchvision.dataset.CIFAR10)
        """

        torch.manual_seed(int(train))

        rad = np.radians(amount)

        s1 = (torch.rand(len(self.data)) * 2 - 1) * amount

        zero = s1 * 0.0
        one = zero + 1.0

        matrices = torch.stack((torch.stack((torch.exp(s1), zero, zero), 0),
                                torch.stack((zero, torch.exp(s1), zero), 0)), 0).permute(2, 0, 1)

        xmin, xmax = self.data.min(), self.data.max()
        data_dtype = self.data.dtype

        self.data = torch.Tensor(self.data.transpose(0, 3, 1, 2))
        grids = F.affine_grid(matrices, self.data.shape, align_corners=False)

        self.data = F.grid_sample(self.data.float(), grids, align_corners=False, mode=mode)
        self.data = self.data.clamp(xmin, xmax).numpy().astype(data_dtype)
        self.data = self.data.transpose(0, 2, 3, 1)


class RotatedCIFAR100(datasets.CIFAR100):
    """ CIFAR100 rotated by fixed amount using random seed """

    def __init__(self, root: str, degree: float, mode: str = 'bilinear',
                 train: bool = True, transform: Union[Callable, type(None)] = None,
                 target_transform: Union[Callable, type(None)] = None, download: bool = False):
        super().__init__(root, train, transform, target_transform, download)
        """
        Args:
            root (string): Root directory of dataset where ``CIFAR100/processed/training.pt``
                and  ``CIFAR100/processed/test.pt`` exist.
            degree (float): Amount of rotation in degrees
            mode (float): Mode used for interpolation (nearest|bilinear|bicubic)
            train (bool, optional): If True, creates dataset from ``training.pt``,
                otherwise from ``test.pt``.
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
                same arguments as torchvision.see arguments of torchvision.dataset.CIFAR100)
        """

        torch.manual_seed(int(train))

        rad = np.radians(degree)

        thetas = (torch.rand(len(self.data)) * 2 - 1) * rad

        c, s = torch.cos(thetas), torch.sin(thetas)

        rot_matrices = torch.stack((torch.stack((c, -s, c*0), 0),
                                    torch.stack((s, c, s*0), 0)), 0).permute(2, 0, 1)

        xmin, xmax = self.data.min(), self.data.max()
        data_dtype = self.data.dtype

        self.data = torch.Tensor(self.data.transpose(0, 3, 1, 2))
        rot_grids = F.affine_grid(rot_matrices, self.data.shape, align_corners=False)

        self.data = F.grid_sample(self.data.float(), rot_grids, align_corners=False, mode=mode)
        self.data = self.data.clamp(xmin, xmax).numpy().astype(data_dtype)
        self.data = self.data.transpose(0, 2, 3, 1)


class TranslatedCIFAR100(datasets.CIFAR100):
    """ CIFAR100 translated by fixed amount using random seed """

    def __init__(self, root: str, amount: float = 8.0, mode: str = 'bilinear',
                 train: bool = True, transform: Union[Callable, type(None)] = None,
                 target_transform: Union[Callable, type(None)] = None, download: bool = False):
        super().__init__(root, train, transform, target_transform, download)
        """
        Args:
            root (string): Root directory of dataset where ``CIFAR100/processed/training.pt``
                and  ``CIFAR100/processed/test.pt`` exist.
            amount (float): Amount of translation in pixels
            mode (float): Mode used for interpolation (nearest|bilinear|bicubic)
            train (bool, optional): If True, creates dataset from ``training.pt``,
                otherwise from ``test.pt``.
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
                same arguments as torchvision.see arguments of torchvision.dataset.CIFAR100)
        """

        torch.manual_seed(int(train))

        rad = np.radians(amount)

        r1 = (torch.rand(len(self.data)) * 2 - 1) * amount / 14.0
        r2 = (torch.rand(len(self.data)) * 2 - 1) * amount / 14.0

        zero = r1 * 0.0
        one = zero + 1.0

        matrices = torch.stack((torch.stack((one, zero, r1), 0),
                                torch.stack((zero, one, r2), 0)), 0).permute(2, 0, 1)

        xmin, xmax = self.data.min(), self.data.max()
        data_dtype = self.data.dtype

        self.data = torch.Tensor(self.data.transpose(0, 3, 1, 2))
        grids = F.affine_grid(matrices, self.data.shape, align_corners=False)

        self.data = F.grid_sample(self.data.float(), grids, align_corners=False, mode=mode)
        self.data = self.data.clamp(xmin, xmax).numpy().astype(data_dtype)
        self.data = self.data.transpose(0, 2, 3, 1)


class ScaledCIFAR100(datasets.CIFAR100):
    """ CIFAR100 scaled by fixed amount using random seed """

    def __init__(self, root: str, amount: float = LOG2, mode: str = 'bilinear',
                 train: bool = True, transform: Union[Callable, type(None)] = None,
                 target_transform: Union[Callable, type(None)] = None, download: bool = False):
        super().__init__(root, train, transform, target_transform, download)
        """
        Args:
            root (string): Root directory of dataset where ``CIFAR100/processed/training.pt``
                and  ``CIFAR100/processed/test.pt`` exist.
            amount (float): Amount of translation in pixels
            mode (float): Mode used for interpolation (nearest|bilinear|bicubic)
            train (bool, optional): If True, creates dataset from ``training.pt``,
                otherwise from ``test.pt``.
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
                same arguments as torchvision.see arguments of torchvision.dataset.CIFAR100)
        """

        torch.manual_seed(int(train))

        rad = np.radians(amount)

        s1 = (torch.rand(len(self.data)) * 2 - 1) * amount

        zero = s1 * 0.0
        one = zero + 1.0

        matrices = torch.stack((torch.stack((torch.exp(s1), zero, zero), 0),
                                torch.stack((zero, torch.exp(s1), zero), 0)), 0).permute(2, 0, 1)

        xmin, xmax = self.data.min(), self.data.max()
        data_dtype = self.data.dtype

        self.data = torch.Tensor(self.data.transpose(0, 3, 1, 2))
        grids = F.affine_grid(matrices, self.data.shape, align_corners=False)

        self.data = F.grid_sample(self.data.float(), grids, align_corners=False, mode=mode)
        self.data = self.data.clamp(xmin, xmax).numpy().astype(data_dtype)
        self.data = self.data.transpose(0, 2, 3, 1)


def get_dataset(dataset, data_root, download_data, transform):
    if dataset == 'mnist':
        train_dataset = RotatedMNIST(data_root, 0, train=True, download=download_data, transform=transform)
        test_dataset = RotatedMNIST(data_root, 0, train=False, download=download_data, transform=transform)
    elif dataset == 'mnist_r90':
        train_dataset = RotatedMNIST(data_root, 90, train=True, download=download_data, transform=transform)
        test_dataset = RotatedMNIST(data_root, 90, train=False, download=download_data, transform=transform)
    elif dataset == 'mnist_r180':
        train_dataset = RotatedMNIST(data_root, 180, train=True, download=download_data, transform=transform)
        test_dataset = RotatedMNIST(data_root, 180, train=False, download=download_data, transform=transform)
    elif dataset == 'translated_mnist':
        train_dataset = TranslatedMNIST(data_root, 8, train=True, download=download_data, transform=transform)
        test_dataset = TranslatedMNIST(data_root, 8, train=False, download=download_data, transform=transform)
    elif dataset == 'scaled_mnist':
        train_dataset = ScaledMNIST(data_root, np.log(2), train=True, download=download_data, transform=transform)
        test_dataset = ScaledMNIST(data_root, np.log(2), train=False, download=download_data, transform=transform)
    elif dataset == 'fmnist':
        train_dataset = RotatedFashionMNIST(data_root, 0, train=True, download=download_data, transform=transform)
        test_dataset = RotatedFashionMNIST(data_root, 0, train=False, download=download_data, transform=transform)
    elif dataset == 'fmnist_r90':
        train_dataset = RotatedFashionMNIST(data_root, 90, train=True, download=download_data, transform=transform)
        test_dataset = RotatedFashionMNIST(data_root, 90, train=False, download=download_data, transform=transform)
    elif dataset == 'fmnist_r180':
        train_dataset = RotatedFashionMNIST(data_root, 180, train=True, download=download_data, transform=transform)
        test_dataset = RotatedFashionMNIST(data_root, 180, train=False, download=download_data, transform=transform)
    elif dataset == 'translated_fmnist':
        train_dataset = TranslatedFashionMNIST(data_root, 8, train=True, download=download_data, transform=transform)
        test_dataset = TranslatedFashionMNIST(data_root, 8, train=False, download=download_data, transform=transform)
    elif dataset == 'scaled_fmnist':
        train_dataset = ScaledFashionMNIST(data_root, np.log(2), train=True, download=download_data, transform=transform)
        test_dataset = ScaledFashionMNIST(data_root, np.log(2), train=False, download=download_data, transform=transform)
    elif dataset == 'cifar10':
        train_dataset = RotatedCIFAR10(data_root, 0, train=True, download=download_data, transform=transform)
        test_dataset = RotatedCIFAR10(data_root, 0, train=False, download=download_data, transform=transform)
    elif dataset == 'cifar10_r90':
        train_dataset = RotatedCIFAR10(data_root, 90, train=True, download=download_data, transform=transform)
        test_dataset = RotatedCIFAR10(data_root, 90, train=False, download=download_data, transform=transform)
    elif dataset == 'cifar10_r180':
        train_dataset = RotatedCIFAR10(data_root, 180, train=True, download=download_data, transform=transform)
        test_dataset = RotatedCIFAR10(data_root, 180, train=False, download=download_data, transform=transform)
    elif dataset == 'translated_cifar10':
        train_dataset = TranslatedCIFAR10(data_root, 8, train=True, download=download_data, transform=transform)
        test_dataset = TranslatedCIFAR10(data_root, 8, train=False, download=download_data, transform=transform)
    elif dataset == 'scaled_cifar10':
        train_dataset = ScaledCIFAR10(data_root, np.log(2), train=True, download=download_data, transform=transform)
        test_dataset = ScaledCIFAR10(data_root, np.log(2), train=False, download=download_data, transform=transform)
    elif dataset == 'cifar100':
        train_dataset = RotatedCIFAR100(data_root, 0, train=True, download=download_data, transform=transform)
        test_dataset = RotatedCIFAR100(data_root, 0, train=False, download=download_data, transform=transform)
    elif dataset == 'cifar100_r90':
        train_dataset = RotatedCIFAR100(data_root, 90, train=True, download=download_data, transform=transform)
        test_dataset = RotatedCIFAR100(data_root, 90, train=False, download=download_data, transform=transform)
    elif dataset == 'cifar100_r180':
        train_dataset = RotatedCIFAR100(data_root, 180, train=True, download=download_data, transform=transform)
        test_dataset = RotatedCIFAR100(data_root, 180, train=False, download=download_data, transform=transform)
    elif dataset == 'translated_cifar100':
        train_dataset = TranslatedCIFAR100(data_root, 8, train=True, download=download_data, transform=transform)
        test_dataset = TranslatedCIFAR100(data_root, 8, train=False, download=download_data, transform=transform)
    elif dataset == 'scaled_cifar100':
        train_dataset = ScaledCIFAR100(data_root, np.log(2), train=True, download=download_data, transform=transform)
        test_dataset = ScaledCIFAR100(data_root, np.log(2), train=False, download=download_data, transform=transform)
    elif dataset == 'tinyimagenet':
        train_dataset = ImageFolder(data_root + '/tiny-imagenet-200/train', transform=transform)
        test_dataset = ImageFolder(data_root + '/tiny-imagenet-200/val', transform=transform)
    elif dataset == 'imagenet':
        pass
    else:
        raise NotImplementedError(f'Unknown dataset: {dataset}')
    return train_dataset, test_dataset
