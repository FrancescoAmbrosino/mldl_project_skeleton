import torch
from torch import nn

from models.customnet import CustomNet
from dataset.tiny_imagenet import get_dataloaders

# Best weights saved by train.py (relative to the repo root, git-ignored)
CHECKPOINT_PATH = 'checkpoints/customnet_best.pth'


# Validation loop
def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0

    correct, total = 0, 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss = val_loss / len(val_loader)
    val_accuracy = 100. * correct / total

    print(f'\nValidation - Loss: {val_loss:.6f} Acc: {val_accuracy:.2f}%')
    return val_loss, val_accuracy


# Runs only with `python eval.py`: train.py imports validate without evaluating anything
if __name__ == '__main__':
    model = CustomNet().cuda()
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    print(f'Loaded weights from {CHECKPOINT_PATH}')

    _, val_loader = get_dataloaders()
    criterion = nn.CrossEntropyLoss()

    validate(model, val_loader, criterion)
