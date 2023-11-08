
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

logger = get_logger(logpath="scl_logs.txt")

def train_epoch_scl(encoder, projector, train_loader, criterion, optimizer, scheduler):

    epoch_loss = 0.0

    for data, target in train_loader:

        with torch.no_grad():
            data_t1 = data.to(args.device)
            data_t2 = data.to(args.device)

        feat1, feat2 = encoder(data_t1), encoder(data_t2)
        proj1, proj2 = projector(feat1), projector(feat2)
        #target = torch.nn.functional.one_hot(target, num_classes=10).float()
        loss = criterion(proj1, proj2, target)

        epoch_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss = epoch_loss / len(train_loader)
    scheduler.step()
    
    return epoch_loss

def train_scl(encoder, projector, train_loader, criterion, optimizer, scheduler, epochs):
    logger.info(encoder)
    logger.info(projector)

    best_loss = None
    train_losses = []
    best_state = {}
    encoder.train()
    projector.train()

    for i in range(1, epochs+1):

        logger.info(f"Epoch {i}")
        train_loss = train_epoch_scl(encoder, projector, train_loader, criterion, optimizer, scheduler)
        logger.info(f"Current Train Loss : {format(train_loss, '.4f')}")
        train_losses.append(train_loss)  

        if best_loss is None:
            best_loss = train_loss
        if best_loss > train_loss:
            best_loss = train_loss
            best_state = {"encoder": encoder.state_dict()}
            
    logger.info(f"Last Loss : {format(train_loss, '.4f')}\tBest Loss : {format(best_loss, '.4f')}")

    return train_losses, encoder, best_state

def linear_train_epoch(encoder, classifier, train_loader, criterion, optimizer):

    epoch_loss = 0.0
    right_number = 0

    classifier.train()

    for data, target in train_loader:
        data, target = data.to(args.device), target.to(args.device)  

        with torch.no_grad():
            features = encoder(data)
        
        optimizer.zero_grad()

        output = classifier(features)
        target = torch.nn.functional.one_hot(target, num_classes=10).float()
        loss = criterion(output, target)
            
        epoch_loss += loss.item()

        predicted = torch.max(output.data,1)[1]
        labeled = torch.max(target.data,1)[1]
        right_number += (predicted == labeled).sum()
    
        loss.backward()
        optimizer.step()

    train_acc=right_number/len(train_loader.dataset)
    epoch_loss = epoch_loss / len(train_loader)

    return epoch_loss, train_acc

def linear_eval_epoch(encoder, classifier, val_loader, criterion):

    epoch_loss = 0.0

    right_number = 0

    classifier.eval()
    encoder.eval()

    with torch.no_grad():

        for data, target in val_loader:
            data, target = data.to(args.device), target.to(args.device)
            
            output = classifier(encoder(data))
            target = torch.nn.functional.one_hot(target, num_classes=10).float()
            loss = criterion(output, target)
            epoch_loss += loss.item()

            predicted = torch.max(output.data,1)[1]
            labeled = torch.max(target.data,1)[1]
            right_number += (predicted == labeled).sum()

    epoch_loss = epoch_loss / len(val_loader)
    val_acc=right_number/len(val_loader.dataset)

    return epoch_loss, val_acc

def linear_scl(encoder, checkpoint, classifier, train_loader, val_loader, criterion, optimizer, epochs):
    logger.info(encoder)
    logger.info(classifier)
    train_losses = []; val_losses = [];  train_acc_scores = [] ;val_acc_scores = []

    best_val_acc = 0

    state_dict = checkpoint["encoder"]
    encoder.load_state_dict(state_dict)

    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()

    for i in range(1, epochs+1):

        logger.info(f"Epoch {i}")

        train_loss, train_acc = linear_train_epoch(encoder, classifier, train_loader, criterion, optimizer)
        train_losses.append(train_loss);  train_acc_scores.append(train_acc)
        logger.info(f"Train loss : {format(train_loss, '.4f')}\tTrain Acc : {format(train_acc, '.4f')}")

        val_loss,  val_acc = linear_eval_epoch(encoder, classifier, val_loader, criterion)
        val_losses.append(val_loss);  val_acc_scores.append(val_acc)
        logger.info(f"Val loss : {format(val_loss, '.4f')}\tVal Acc : {format(val_acc, '.4f')}")

        if best_val_acc < val_acc:
            best_epoch_acc = i
            best_val_acc = val_acc
        
    logger.info(f"best acc is {format(best_val_acc, '.4f')} at epoch {best_epoch_acc}")

    return train_losses, val_losses, train_acc_scores, val_acc_scores