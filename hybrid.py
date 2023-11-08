
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


def train_epoch_supconce(encoder, projector, classifier, train_loader, criterion, optimizer, scheduler):

    epoch_loss = 0.0
    
    right_number = 0

    encoder.train()
    classifier.train()
    projector.train()

    for data, target in train_loader:
        data, target = data.to(args.device), target.to(args.device)

        with torch.no_grad():
            data_t1 = data
            data_t2 = data
      
        optimizer.zero_grad()

        feature1, feature2 = encoder(data_t1), encoder(data_t2)
        prediction1, prediction2 = classifier(feature1), classifier(feature2)
        projection1, projection2 = projector(feature1), projector(feature2)
        #target = torch.nn.functional.one_hot(target, num_classes=10).float()
        loss = criterion(projection1, projection2, prediction1, prediction2, target)         
        epoch_loss += loss.item()

        targets = torch.cat([target, target], dim=0)
        predictions = torch.cat([prediction1, prediction2], dim=0)

        _, labels_predicted = torch.max(predictions, dim=1)
        labeled = torch.max(targets.data,1)[1]
        
        right_number += (labels_predicted == labeled).sum()
        
        loss.backward()
        optimizer.step()

    scheduler.step()

    epoch_loss = epoch_loss / len(train_loader)
    train_acc=right_number/len(train_loader.dataset)
    return epoch_loss, train_acc

def val_epoch_supconce(encoder, classifier, val_loader, criterion_sl):

    epoch_loss = 0.0

    right_number = 0

    encoder.eval()
    classifier.eval()

    with torch.no_grad():

        for data, target in val_loader:
            data_t, target = data.to(args.device), target.to(args.device)
            
            features = encoder(data_t)
            predictions = classifier(features)
            target = torch.nn.functional.one_hot(target, num_classes=10).float()
            loss = criterion_sl(predictions, target)
            epoch_loss += loss.item()

            _, labels_predicted = torch.max(predictions, dim=1)

            labeled = torch.max(target.data,1)[1]
        
            right_number += (labels_predicted == labeled).sum()
            

    epoch_loss = epoch_loss / len(val_loader)
    val_acc=right_number/len(val_loader.dataset)

    return epoch_loss, val_acc

def train_supconce(encoder, projector, classifier, train_loader, val_loader, criterion, criterion_sl, optimizer, epochs, scheduler):

    logger = get_logger(logpath="hybrid_logs.txt")
    logger.info(encoder)
    logger.info(classifier)
    logger.info(projector)

    train_losses = []; val_losses = []; train_acc_scores = [] ; val_acc_scores = []

    best_val_acc = 0
    best_epoch =0 
    for i in range(1, epochs+1):
        
        logger.info(f"Epoch {i}")
        train_loss,train_acc = train_epoch_supconce(encoder, projector, classifier, train_loader, criterion, optimizer, scheduler)
        train_losses.append(train_loss); train_acc_scores.append(train_acc)
        logger.info(f"Train loss : {format(train_loss, '.4f')}\tTrain Acc : {format(train_acc, '.4f')}")
        val_loss, val_acc = val_epoch_supconce(encoder, classifier, val_loader, criterion_sl)
        val_losses.append(val_loss); val_acc_scores.append(val_acc)
        logger.info(f"Val loss : {format(val_loss, '.4f')}\tVal Acc : {format(val_acc, '.4f')}")            

        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_epoch=i

    logger.info(f"best acc is {format(best_val_acc, '.4f')} at epoch {best_epoch}")

    return train_losses, val_losses, train_acc_scores, val_acc_scores