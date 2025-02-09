import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torch.nn import functional as F
from dataloader import ClassIncrementalSampler
from buffer import Buffer
from model import ResNet18
from training import training_gs
from dataset import get_benchmarks, get_training_validation_datasets, input_size


### FUNZIONI PER LE COMPONENTI DEL METODO

def prepare_datasets(cifar10=False, cifar100=False):
    if cifar10 and cifar100:
        benchmark_cifar10, benchmark_cifar100 = get_benchmarks(cifar10, cifar100)
        train_dataset_10, val_dataset_10 = get_training_validation_datasets(benchmark_cifar10)
        train_dataset_100, val_dataset_100 = get_training_validation_datasets(benchmark_cifar100)
        return (train_dataset_10, val_dataset_10), (train_dataset_100, val_dataset_100)
    if cifar10:
        benchmark_cifar10 = get_benchmarks(cifar10, cifar100)
        train_dataset, val_dataset = get_training_validation_datasets(benchmark_cifar10)
        return train_dataset, val_dataset
    if cifar100:
        benchmark_cifar100 = get_benchmarks(cifar10, cifar100)
        train_dataset, val_dataset = get_training_validation_datasets(benchmark_cifar100)
        return train_dataset, val_dataset
    else:
        return None


def get_loaders(TR, VL, batch_size_tr, batch_size_vl, n_tasks):
    train_sampler = ClassIncrementalSampler(TR, n_tasks)
    train_loader  = torch.utils.data.DataLoader(
            TR,
            num_workers=2,
            sampler=train_sampler,
            batch_size=batch_size_tr,
            pin_memory=True
        )

    val_sampler = ClassIncrementalSampler(VL, n_tasks)
    val_loader  = torch.utils.data.DataLoader(
            VL,
            num_workers=2,
            sampler=val_sampler,
            batch_size=batch_size_vl,
            pin_memory=True
        )

    return train_loader, val_loader


def get_buffer(capacity, n_classes, x_size, y_size, x_dtype, y_dtype, device):
    mem_size = capacity * n_classes
    buffer = Buffer(capacity=mem_size, device=device)
    buffer.add_buffer('bx', x_dtype, x_size)
    buffer.add_buffer('by', y_dtype, y_size)

    return buffer


### FUNZIONI PER VALUTARE I COSTI

def count_flops(n_classes, device, verbose):
    model = ResNet18(nclasses=n_classes,
                    nf=20,
                    input_size=input_size,
                    dist_linear='ace')
    model.to(device)
    
    flops = model.one_sample_flop(device)
    if verbose: print(f'FLOPs per sample: {flops}')
    return flops


def model_bits_occupacy(n_classes, verbose):
    model = ResNet18(nclasses=n_classes,
                    nf=20,
                    input_size=input_size,
                    dist_linear='ace')
    n_params = sum(np.prod(p.size()) for p in model.parameters())
    bits = n_params * 32
    if verbose: print(f"Bits used for the model: {bits}")
    return bits


### FUNZIONI PER EFFETTUARE I TEST

def print_and_get_results(eval_accs, buffer):
    
    
    # Final Results

    accs = np.stack(eval_accs).T
    avg_acc = accs[:, -1].mean()
    avg_fgt = (accs.max(1) - accs[:, -1])[:-1].mean() #average forgetting
    avg_accuracies = accs.T.sum(axis=1) / np.count_nonzero(accs.T, axis=1)
    aaa = avg_accuracies.mean()

    print('\nFinal Results\n')
    print(f'avg_acc: {avg_acc:.4f}')
    print(f'avg_fgt: {avg_fgt:.4f}')
    print(f'avg_anytime_acc: {aaa:.4f}')
    print(f'metrics: buffer_n_bits: {buffer.n_bits()}')

    return avg_acc, avg_fgt, aaa


def experiment(M, train_dataset, val_dataset, batch_size, rehearsal_batch_size, n_tasks, aug_flag, device, verbose):
    train_loader, val_loader = get_loaders(train_dataset, val_dataset, batch_size, 128, n_tasks)

    n_classes = len(np.unique(val_dataset.tensors[1]))
    model = ResNet18(nclasses=n_classes,
                    nf=20,
                    input_size=input_size,
                    dist_linear='ace')

    buffer = get_buffer(capacity=M,
                        n_classes=n_classes,
                        x_size=val_dataset.tensors[0][0].shape, # [3, 32, 32]
                        y_size=val_dataset.tensors[1][0].shape, # []
                        x_dtype=val_dataset.tensors[0].dtype, # torch.float32
                        y_dtype=val_dataset.tensors[1].dtype, # torch.int64
                        device=device)

    eval_accs, model = training_gs(original_model=model,
                                   criterion=F.cross_entropy,
                                   n_tasks=n_tasks,
                                   train_loader=train_loader,
                                   val_loader=val_loader,
                                   buffer=buffer,
                                   re_batch_size=rehearsal_batch_size,
                                   aug_flag=aug_flag,
                                   device=device,
                                   verbose=verbose)

    return print_and_get_results(eval_accs, buffer), eval_accs


### FUNZIONI PER PLOTTARE I RISULTATI

def plot_results(M, accuracies, figsize=(7, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    colormap = cm.Blues(np.linspace(0.3, 1.0, len(accuracies[0])))
    avg_accuracies = accuracies.T.sum(axis=1) / np.count_nonzero(accuracies.T, axis=1)

    for i, row in enumerate(accuracies):
        ax.plot(row, color=colormap[i], marker="o", label=f"Accuracy on task {i}")
    ax.plot(avg_accuracies, color="red", marker='o', label=f"Average Accuracy")

    ax.set_title(f"Accuracies plots M={M}", fontsize=14)
    ax.set_xlabel("Experience", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.7)

    plt.show()


def plot_aa(M, accuracies, figsize=(7, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    avg_accuracies = accuracies.T.sum(axis=1) / np.count_nonzero(accuracies.T, axis=1)
    ax.plot(avg_accuracies, color="red", marker='o', label=f"Average Accuracy")

    ax.set_title(f"Average Accuracy plot M={M}", fontsize=14)
    ax.set_xlabel("Experience", fontsize=12)
    ax.set_ylabel("Average Accuracy", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.7)

    plt.show()
