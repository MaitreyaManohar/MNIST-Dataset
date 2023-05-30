import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


def viewPlots(train_loader):
    fig = plt.figure()
    batch_data, batch_label = next(iter(train_loader))
    for i in range(12):
        plt.subplot(3, 4, i + 1)
        plt.tight_layout()
        plt.imshow(batch_data[i].squeeze(0), cmap="gray")
        plt.title(batch_label[i].item())
        plt.xticks([])
        plt.yticks([])


def getTrainLoader():
    batch_size = 512
    train_transforms = transforms.Compose(
        [
            transforms.RandomApply(
                [
                    transforms.CenterCrop(22),
                ],
                p=0.1,
            ),
            transforms.Resize((28, 28)),
            transforms.RandomRotation((-15.0, 15.0), fill=0),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": 2,
        "pin_memory": True,
    }
    train_data = datasets.MNIST(
        "../data", train=True, download=True, transform=train_transforms
    )
    train_loader = torch.utils.data.DataLoader(train_data, **kwargs)
