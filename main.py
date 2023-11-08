import os 
import torch
import torch.nn as nn
from models import ResNet18, Projector, LinearClassifier
from losses import SupConLoss, SupConCELoss
from ce import train_ce
from hybrid import train_supconce
from scl import train_scl, linear_scl
from args import args
from data import train_loader,val_loader

METHOD = args.method

DEFAULT_NUM_CLASSES = 10 #for cross entropy
DEFAULT_OUT_DIM = 128  #for ssl embedding space dimension

# Model definition
if args.method == 'sl':
    embed_only = False
else:
    embed_only = True
    projector = Projector(name=args.backbone, out_dim=DEFAULT_OUT_DIM, device=args.device)
    classifier = LinearClassifier(name=args.backbone, num_classes=DEFAULT_NUM_CLASSES, device=args.device)

if args.backbone == 'resnet18':
    PATH_TO_WEIGHTS = 'pnn/resnet18.pth'
    model = ResNet18(num_classes=DEFAULT_NUM_CLASSES, embed_only=embed_only, from_scratch=args.scratch, path_to_weights=PATH_TO_WEIGHTS, device=args.device)

'''
根据损失函数选择优化器
sl为交叉熵损失
scl为有监督对比损失
hybrid为交叉熵损失和有监督对比损失结合

我们使用余弦退火作为学习速率衰减方案,不使用热重启
除了监督对比的第二阶段,我们使用0.1的学习速率在固定表示上训练线性分类器，而不需要调度
'''
### OPTIMISER
if METHOD == 'sl':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
elif METHOD == 'scl':
    optimizer = torch.optim.Adam(list(model.parameters()) + list(projector.parameters()), lr=args.lr, weight_decay=args.wd) 
    optimizer2 = torch.optim.Adam(classifier.parameters(), lr=args.lr2, weight_decay=args.wd)
elif METHOD == 'hybrid':
    optimizer = torch.optim.Adam(list(model.parameters()) + list(classifier.parameters()) + list(projector.parameters()), lr=args.lr, weight_decay=args.wd)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-6) 

criterion_ce = nn.CrossEntropyLoss()

if METHOD == 'sl':
    history = train_ce(model, train_loader, val_loader, criterion_ce, optimizer, args.epochs, scheduler)
    del model

elif METHOD == 'scl':
    criterion = SupConLoss(temperature=args.tau, device=args.device)
    ssl_train_losses, model, last_checkpoint = train_scl(model, projector, train_loader, criterion, optimizer, scheduler, args.epochs)
    history = linear_scl(model, last_checkpoint ,classifier, train_loader, val_loader, criterion_ce, optimizer2, args.epochs2)
    del model; del projector; del classifier
    
elif METHOD == 'hybrid':
    criterion = SupConCELoss(temperature=args.tau, alpha=args.alpha, device=args.device)
    history = train_supconce(model, projector, classifier, train_loader, val_loader, criterion, criterion_ce, optimizer, args.epochs, scheduler)
    del model; del projector; del classifier

del train_loader; del val_loader
