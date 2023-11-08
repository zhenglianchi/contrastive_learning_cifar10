import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义基本的卷积块
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out



# 定义ResNet模型
class ResNet(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks=[2,2,2,2], num_classes=10,embed_only=False):
        super(ResNet, self).__init__()
        self.embed_only=embed_only
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self.make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self.make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self.make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        if self.embed_only:
            return out
        out = self.fc(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes=10,  embed_only=False, from_scratch=False, path_to_weights="pnn/resnet18.pth", device="cuda"):
        super(ResNet18, self).__init__()

        self.resnet18 = ResNet(num_classes=num_classes,  embed_only=embed_only).to(device)
        if not from_scratch:
            weights = torch.load(path_to_weights, map_location=device)
            state_dict = {k: v for k, v in weights.items() if k in self.resnet18.state_dict().keys() and k!="fc.weight" and k!="fc.bias"}
            self.resnet18.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        return self.resnet18(x)


'''
为所述SCL和所述混合设置的第二头部添加一个具有一个隐层、128个神经元的MLP
'''
class Projector(nn.Module):
    def __init__(self, name='resnet18', out_dim=128, apply_bn=False, device="gpu"):
        super(Projector, self).__init__()
        _, dim_in = model_dict[name]
        self.linear1 = nn.Linear(dim_in, dim_in)
        self.linear2 = nn.Linear(dim_in, out_dim)
        self.bn = nn.BatchNorm1d(dim_in)
        self.relu = nn.ReLU()
        if apply_bn:
            self.projector = nn.Sequential(self.linear1, self.bn, self.relu, self.linear2)
        else:
            self.projector = nn.Sequential(self.linear1, self.relu, self.linear2)
        self.projector = self.projector.to(device)

    def forward(self, x):
        return self.projector(x)


'''
线性分类器
'''
class LinearClassifier(nn.Module):
    def __init__(self, name='resnet18', num_classes=10, device="cuda"):
        super(LinearClassifier, self).__init__()
        _, feat_dim = model_dict[name]
        self.fc = nn.Linear(feat_dim, num_classes).to(device)

    def forward(self, features):
        return self.fc(features)
    
model_dict = {
    'resnet18' : ["resnet18", 512],
}