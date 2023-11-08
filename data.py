from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from args import args

myTransforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomResizedCrop((224,224),scale=(0.8,1.0),ratio=(1.0,1.0)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
testTransforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
cifar_train = datasets.CIFAR10('cifar', True,  transform=myTransforms,  download=True)
cifar_test = datasets.CIFAR10('cifar', False, transform=testTransforms, download=True)
train_loader = DataLoader(dataset=cifar_train, batch_size=args.bs, shuffle=True, drop_last=True)   # 加载数据集
val_loader = DataLoader(dataset=cifar_test, batch_size=args.bs, shuffle=True, drop_last=True)   # 加载数据集