import torch
import torch.nn as nn


class SmallerVGGNet(nn.Module):
    def __init__(self, pretrained=False):
        super(SmallerVGGNet, self).__init__()
        default_input_channels = 3

        self.conv1 = nn.Conv2d(default_input_channels, 32, kernel_size=(3, 3))
        self.bn_1 = nn.BatchNorm2d(32, eps=0.001)
        self.mp_1 = nn.MaxPool2d(3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=(3, 3))

        self.bn_2 = nn.BatchNorm2d(64, eps=0.001)
        self.mp_2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3))
        self.bn_3 = nn.BatchNorm2d(128, eps=0.001)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=(3, 3))

        self.classifier = None

        self.drop0_25 = nn.Dropout(p=0.25)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.bn_1(x)
        x = self.mp_1(x)
        x = self.drop0_25(x)

        x = self.conv2(x)
        x = torch.relu(x)
        x = self.bn_2(x)

        x = self.conv2_2(x)
        x = torch.relu(x)
        x = self.bn_2(x)
        x = self.mp_2(x)
        x = self.drop0_25(x)

        x = self.conv3(x)
        x = torch.relu(x)
        x = self.bn_3(x)

        x = self.conv4(x)
        x = torch.relu(x)
        x = self.bn_3(x)
        x = self.mp_2(x)

        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x













        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Conv2d(64, 128, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Conv2d(128, 256, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Conv2d(256, 512, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2)
        # )



    # def forward(self, x):
    #     x = self.conv2d_1(x)
    #     x = torch.relu(x)
    #     x = self.bn_1(x)
    #     x = self.maxpool3(x)
    #     x = torch.dropout(x, 0.25)
    #     x = self.conv2d_2(x)
    #     x = torch.relu(x)
    #     x = self.bn_1(x)
    #     x = self.conv2d_2(x)
    #     x = torch.relu(x)
    #     x = self.bn_1(x)
    #     x = self.maxpool2(x)
    #     x = torch.dropout(x, 0.25)


