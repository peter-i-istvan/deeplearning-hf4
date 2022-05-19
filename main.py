import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
import matplotlib.pyplot as plt
from classifier import ConvNet
from tqdm import tqdm
import numpy as np
import sys


def train_or_validate(net, epoch, parameters, have_cuda=False, is_training=False):
    """
    "parameters" is always a dictionary:
        - train_loader
        - validation_loader
        - optimizer
        - criterion
    """
    running_loss = 0.0
    correct = 0.0
    total = 0
    if is_training:
        net.train()  # set to training mode
    else:
        net.eval()

    if is_training:
        loader = parameters["train_loader"]
    else:
        loader = parameters["validation_loader"]

    optimizer = parameters["optimizer"]
    criterion = parameters["criterion"]
    for i, data in enumerate(tqdm(loader, file=sys.stdout), 0):
        inputs, labels = data
        if have_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        if is_training:
            loss.backward()
            optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        # w_s = net.get_weights_l2_norm()
        # print(f"After minibatch {i} weights size is {w_s} at average")

    avg_loss = running_loss / len(loader)
    corr_percent = correct / total * 100
    print(
        f"{'Train' if is_training else 'Validation'} epoch {epoch+1} loss: {avg_loss:.3f} correct: {corr_percent:.2f}"
    )
    return avg_loss, corr_percent


def plot_statistics(tr_losses, val_losses, tr_accs, val_accs):
    plt.plot(tr_losses)
    plt.plot(val_losses)
    plt.show()
    plt.plot(tr_accs)
    plt.plot(val_accs)
    plt.show()


def main():
    # Set-up
    have_cuda = torch.cuda.is_available()
    np.random.seed(42)
    torch.manual_seed(42)
    if have_cuda:
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print("CUDA is available")
    else:
        print("CUDA is not available")
    # Create net instance, on GPU if available
    net = ConvNet(8)
    if have_cuda:
        net = net.cuda()
    # Hyperparameters: criterion, optimizer, learning rate scheduler, epoch number
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=1e-4)
    scheduler = lr_scheduler.StepLR(optimizer, 10)
    numEpoch = 3
    # Data, loaders
    train_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])  # TODO: add normalize
    validation_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])  # TODO: add normalize
    train_set = torchvision.datasets.ImageFolder("db/Classification/train/", train_transform)
    validation_set = torchvision.datasets.ImageFolder("db/Classification/val/", validation_transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=128, shuffle=True)
    # Training loop
    tr_losses = []
    tr_accs = []
    val_losses = []
    val_accs = []
    parameters = {
        "train_loader": train_loader,
        "validation_loader": validation_loader,
        "optimizer": optimizer,
        "criterion": criterion
    }
    for epoch in range(numEpoch):
        tr_loss, tr_corr = train_or_validate(net, epoch, parameters, have_cuda, is_training=True)
        val_loss, val_corr = train_or_validate(net, epoch, parameters, have_cuda, is_training=False)
        tr_losses.append(tr_loss)
        tr_accs.append(tr_corr)
        val_losses.append(val_loss)
        val_accs.append(val_corr)
        scheduler.step()

    plot_statistics(tr_losses, val_losses, tr_accs, val_accs)


if __name__ == '__main__':
    main()

