import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F



class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


class SqueezeNet(nn.Module):

    def __init__(self, version='1_0', num_classes=1000):
        super(SqueezeNet, self).__init__()
        self.num_classes = num_classes
        self.maxpool1_8 = nn.Sequential(
            nn.MaxPool2d(kernel_size=16, stride=8, ceil_mode=True),
        )
        self.maxpool1_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=7, stride=4, ceil_mode=True),
        )
        self.maxpool1_2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
        )
        self.deconvolution = nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, stride=2),
        )
        self.fc = torch.nn.Linear(165483, 256)
        self.fc2 = torch.nn.Linear(93312,256)
        if version == '1_0':
            self.features1 = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7, stride=2),
                nn.ReLU(inplace=True),
            )
            self.features2 = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
            )



    def forward(self, x):

        feature_x1 = self.maxpool1_8(x)
        x1 = self.features1(x)
        feature_x2 = self.maxpool1_4(x1)
        x2 = self.features2(x1)
        feature_x3 = self.maxpool1_2(x2)
        #feature_x3_1 = torch.mean(feature_x3,3)
        #print(feature_x3.size())
        #print(feature_x3_1.size())
        x_3 = torch.reshape(feature_x3,(1,93312))
        output_x = self.fc2(x_3)
        output_x = F.softmax(output_x, dim=0)
        print(output_x)
        testx_1 = torch.cat((feature_x1,feature_x2),dim=1)
        testx_2 = torch.cat((testx_1,feature_x3),dim=1)
        Flatten_x = torch.flatten(testx_2)
        Flatten_x = torch.unsqueeze(Flatten_x,0)
        #print(Flatten_x)
        output_x1 = self.fc(Flatten_x)
        output_x1 = F.softmax(output_x1,dim=0)
        output_x2 = self.fc(Flatten_x)
        output_x2 = F.softmax(output_x2,dim=0)
        output_x3 = self.fc(Flatten_x)
        output_x3 = F.softmax(output_x3,dim=0)
        #print(output_x1)
        #return output_x1,output_x2,output_x3
        return output_x