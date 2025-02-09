from torchvision import datasets
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
import numpy as np
import torch
import kornia

SEED = 42
H = 32 # Size immagini
input_size = (3, H, H)

transform = torch.nn.Sequential(kornia.augmentation.RandomCrop(size=(H, H), padding=4, fill=-1),
                                kornia.augmentation.RandomHorizontalFlip())

def get_benchmarks(cifar10=False, cifar100=False):   

    if cifar10 and cifar100:
        benchmark_cifar10 = datasets.CIFAR10(root='./data', download=True)
        benchmark_cifar10.data = torch.from_numpy(benchmark_cifar10.data).float().permute(0, 3, 1, 2)
        benchmark_cifar10.targets = np.array(benchmark_cifar10.targets)
        benchmark_cifar10.data = (benchmark_cifar10.data / 255. - .5) * 2. # Normalizzazione per avere dati compresi fra -1 e 1 e centrati in 0

        benchmark_cifar100 = datasets.CIFAR100(root='./data', download=True)
        benchmark_cifar100.data = torch.from_numpy(benchmark_cifar100.data).float().permute(0, 3, 1, 2)
        benchmark_cifar100.targets = np.array(benchmark_cifar100.targets)
        benchmark_cifar100.data = (benchmark_cifar100.data / 255. - .5) * 2.

        return benchmark_cifar10, benchmark_cifar100
    if cifar10:
        benchmark_cifar10 = datasets.CIFAR10(root='./data', download=True)
        benchmark_cifar10.data = torch.from_numpy(benchmark_cifar10.data).float().permute(0, 3, 1, 2)
        benchmark_cifar10.targets = np.array(benchmark_cifar10.targets)
        benchmark_cifar10.data = (benchmark_cifar10.data / 255. - .5) * 2.

        return benchmark_cifar10
    if cifar100:
        benchmark_cifar100 = datasets.CIFAR100(root='./data', download=True)
        benchmark_cifar100.data = torch.from_numpy(benchmark_cifar100.data).float().permute(0, 3, 1, 2)
        benchmark_cifar100.targets = np.array(benchmark_cifar100.targets)
        benchmark_cifar100.data = (benchmark_cifar100.data / 255. - .5) * 2.

        return benchmark_cifar100
    else:
        return None

def get_training_validation_datasets(benchmark, val_split=0.05):
    X_train, X_val, y_train, y_val = train_test_split(benchmark.data, benchmark.targets, test_size=val_split, stratify=benchmark.targets, random_state=SEED)

    y_train = torch.from_numpy(y_train)
    y_val = torch.from_numpy(y_val)

    # Ordino il dataset per label
    sorted_indices_train = y_train.argsort()
    sorted_data_train = X_train[sorted_indices_train]
    sorted_labels_train = y_train[sorted_indices_train]

    sorted_indices_val = y_val.argsort()
    sorted_data_val = X_val[sorted_indices_val]
    sorted_labels_val = y_val[sorted_indices_val]

    # Creo i dataset
    train_dataset = TensorDataset(sorted_data_train, sorted_labels_train)
    val_dataset = TensorDataset(sorted_data_val, sorted_labels_val)

    return train_dataset, val_dataset