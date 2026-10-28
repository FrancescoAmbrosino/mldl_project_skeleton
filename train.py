import torch
from torch import nn
from tqdm.auto import tqdm # to have a clue on the remaining training time
import wandb

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

    # Averaged over the epoch while the weights change, on augmented images:
    # not the accuracy the final weights would get on the train set
    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total
    print(f'Train Epoch: {epoch} - Loss: {train_loss:.6f} Acc: {train_accuracy:.2f}%')
    
    return train_loss, train_accuracy

if __name__ == '__main__':
    # The API key comes from the WANDB_API_KEY environment variable (Colab Secrets cell): never write it here
    wandb.init(project='aml-lab3')
    # Hyperparameters live only in the config and the code below reads them from there,
    # so the values wandb shows next to the charts are the ones actually used
    config = wandb.config
    config.learning_rate = 0.01
    config.num_epochs = 10
    config.momentum = 0.9
    config.batch_size = 32  # as in lab 2 run 8, so the results compare
        
    model = CustomNet().cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)
    # Cosine decay from learning_rate to 0 over num_epochs, stepped once per epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=config.num_epochs)

    train_loader, val_loader = get_dataloaders(batch_size=config.batch_size)
    
    best_acc = 0

    # Run the training process for {num_epochs} epochs
    for epoch in range(1, config.num_epochs + 1):
        train_loss, train_accuracy = train(epoch, model, train_loader, criterion, optimizer)

        # At the end of each training iteration, perform a validation step
        val_loss, val_accuracy = validate(model, val_loader, criterion)

        # One log per epoch = one point per epoch on every chart. The lr is read before
        # scheduler.step(), so it is the one used during this epoch
        wandb.log({
            'epoch': epoch,
            'train/loss': train_loss,
            'train/acc': train_accuracy,
            'val/loss': val_loss,
            'val/acc': val_accuracy,
            'lr': optimizer.param_groups[0]['lr'],
        })
        
        # Compare before updating best_acc: afterwards val_accuracy > best_acc is never true
        if val_accuracy > best_acc:
            torch.save(model.state_dict(), CHECKPOINT_PATH)
                    
        # Best validation accuracy
        best_acc = max(best_acc, val_accuracy)

        print(f"lr = {optimizer.param_groups[0]['lr']}\n")
        scheduler.step()

    # By default the summary keeps the last logged value: store the best explicitly
    wandb.summary['best_val_acc'] = best_acc
    wandb.finish()
    print(f'Best validation accuracy: {best_acc:.2f}%')
