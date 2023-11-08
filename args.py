#参数表
import argparse

parser = argparse.ArgumentParser()

#Generic
parser.add_argument("--method", type=str, default='hybrid') #method in ['sl','scl','hybrid']
#parser.add_argument("--device", type=str, default="cuda:0") #device to train on
parser.add_argument("--device", type=str, default="cuda") #device to train on
parser.add_argument("--workers", type=int, default=0) #number of workers
parser.add_argument("--bs", type=int, default=128) #batch size
parser.add_argument("--epochs", type=int, default=50) #nb of epoches
parser.add_argument("--epochs2", type=int, default=50) #nb of epoches for linear classifier of scl and hybrid

#Model
parser.add_argument("--backbone", type=str, default='resnet18')
parser.add_argument("--scratch", action='store_true') #train from scratch

#Optimizer
parser.add_argument("--wd", type=float, default=1e-4) #weight decay
parser.add_argument("--lr", type=float, default=1e-4) #learning rate
parser.add_argument("--lr2", type=float, default=1e-1) #learning rate for linear eval

#Parameter
parser.add_argument("--tau", type=float, default=0.06) #temperature for nt xent loss
parser.add_argument("--alpha", type=float, default=0.5) #tradeoff between cross entropy and nt xent

args = parser.parse_args()