import torch
from torch import nn
from tqdm.auto import tqdm # to have a clue on the remaining training time

from models.customnet import CustomNet
from dataset.tiny_imagenet import get_dataloaders
from eval import validate, CHECKPOINT_PATH


def train(epoch, model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):
        inputs, targets = inputs.cuda(), targets.cuda()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total
    print(f'Train Epoch: {epoch} - Loss: {train_loss:.6f} Acc: {train_accuracy:.2f}%')
    
    return train_loss, train_accuracy


if __name__ == '__main__':
    model = CustomNet().cuda()
    num_epochs = 10

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=num_epochs)

    train_loader, val_loader = get_dataloaders()
    
    best_acc = 0

    # Run the training process for {num_epochs} epochs
    for epoch in range(1, num_epochs + 1):
        train(epoch, model, train_loader, criterion, optimizer)

        # At the end of each training iteration, perform a validation step
        _, val_accuracy = validate(model, val_loader, criterion)

        if val_accuracy > best_acc:
            torch.save(model.state_dict(), CHECKPOINT_PATH)
                    
        # Best validation accuracy
        best_acc = max(best_acc, val_accuracy)

        print(f"lr = {optimizer.param_groups[0]['lr']}")
        scheduler.step()

    print(f'Best validation accuracy: {best_acc:.2f}%')
