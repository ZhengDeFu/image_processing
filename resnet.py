
import torch
from torch import nn
import torch.nn.functional as F

class layer(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1 ,identity_downsample=None):
        super().__init__()
        self.identity_downsample = identity_downsample
        self.expansion = 4
        
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1,stride=1, padding=0)
        self.batch_n1 = nn.BatchNorm2d(out_channels)

        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_n2 = nn.BatchNorm2d(out_channels)

        self.conv_3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_n3 = nn.BatchNorm2d(out_channels*self.expansion)

        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()

        out = self.relu(self.batch_n1(self.conv_1(x)))
        out = self.relu(self.batch_n2(self.conv_2(out)))
        out = self.batch_n3(self.conv_3(out))

        if self.identity_downsample:
            identity = self.identity_downsample(x)

        out = self.relu(out + identity)

        return out



class Resnet(nn.Module):
    
    def __init__(
            self,
            number_layers = [3, 4, 6, 3],
            intermediate_channels_list = [64, 128, 256, 512],
            stride_block = [1, 2, 2, 2],
            num_class = 1000
            
    ):
        super().__init__()
        
        self.in_channels = 64
        blocks = [
            self.make_block(
                layer,
                number_layer ,
                intermediate_channels, stride= stride)
                for number_layer, intermediate_channels, stride in zip(number_layers, intermediate_channels_list, stride_block)
                ] 
        
        self.model = nn.Sequential(
            nn.Conv2d(3,64, 7,2,3, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,2,padding=1),
            *blocks,
            nn.AvgPool2d(1,1),
            nn.Flatten(),
            nn.Linear(7*7*2048,1000)
            # nn.AdaptiveAvgPool2d(1,1),
            # nn.Linear(512*, num_classes)
        )
        
    
    def forward(self, x):
        out = self.model(x)
        # out = self.conv_2(out)
        # out = self.pool_1(out)
        return out 
    
    def make_block(self, make_layer, num_residual_blocks, intermediate_channels, stride):

        identity_downsample = None 
        layers = []

        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(nn.Conv2d(
                self.in_channels,
                intermediate_channels*4,
                kernel_size=1,
                stride=stride,
                bias=False
            ))


        layers.append(make_layer(self.in_channels, intermediate_channels, stride, identity_downsample))

        self.in_channels = intermediate_channels * 4

        for i in range(num_residual_blocks - 1):
            layers.append(make_layer(self.in_channels, intermediate_channels))

        
        return nn.Sequential(*layers)
    
def Resnet50():
    model = Resnet(number_layers=[3, 4, 6, 3])
    return model

def Resnet101():
    model = Resnet(number_layers=[3, 4, 23, 3])
    return model

def Resnet152():
    model = Resnet(number_layers=[3, 8, 36, 3])
    return model

def test_resnet():
    model = Resnet152()
    x = torch.randn(4, 3, 224, 224)
    out = model(x)
    print(out.shape)
    print(model)

test_resnet()