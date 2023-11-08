
import torch
from args import args
import logging

def get_logger(logpath, displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)

    return logger


def train_epoch(model, train_loader,  criterion, optimizer, scheduler):

    epoch_loss = 0.0
    right_number = 0
    model.train()

    for data, target in train_loader:
        data_t, target = data.to(args.device), target.to(args.device)
        
        optimizer.zero_grad()

        output = model(data_t)
        target = torch.nn.functional.one_hot(target, num_classes=10).float()
        loss = criterion(output, target)
    
        epoch_loss += loss.item()
        
        predicted = torch.max(output.data,1)[1]
        labeled = torch.max(target.data,1)[1]
        right_number += (predicted == labeled).sum()

        loss.backward()
        optimizer.step()

    scheduler.step()
    train_acc=right_number/len(train_loader.dataset)
    epoch_loss = epoch_loss / len(train_loader)

    return epoch_loss, train_acc

def val_epoch(model, val_loader, criterion):

    epoch_loss = 0.0
    right_number = 0

    model.eval()

    with torch.no_grad():

        for data, target in val_loader:
            data, target = data.to(args.device), target.to(args.device)
            
            output = model(data)
            target = torch.nn.functional.one_hot(target, num_classes=10).float()
            loss = criterion(output, target)
            epoch_loss += loss.item()

            predicted = torch.max(output.data,1)[1]
            labeled = torch.max(target.data,1)[1]
            right_number += (predicted == labeled).sum()


    val_acc=right_number/len(val_loader.dataset)
    epoch_loss = epoch_loss / len(val_loader)

    return epoch_loss, val_acc

def train_ce(model, train_loader, val_loader, criterion, optimizer, epochs, scheduler):
    logger = get_logger(logpath="ce_logs.txt")
    logger.info(model)

    train_losses = []; val_losses = []; train_accs = []; val_accs = []

    best_val_acc = 0
    best_epoch=0
    for i in range(1, epochs+1):
        
        logger.info(f"Epoch {i}")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scheduler)
        train_losses.append(train_loss);  train_accs.append(train_acc)
        logger.info(f"Train loss : {format(train_loss, '.4f')}\tTrain Acc : {format(train_acc, '.4f')}")

        val_loss, val_acc = val_epoch(model, val_loader, criterion)
        val_losses.append(val_loss);  val_accs.append(val_acc)
        logger.info(f"Val loss : {format(val_loss, '.4f')}\tVal Acc : {format(val_acc, '.4f')}")          
        
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_epoch = i

    logger.info(f"best acc is {format(best_val_acc, '.4f')} at epoch {best_epoch}")

    return train_losses, val_losses, train_accs,val_accs