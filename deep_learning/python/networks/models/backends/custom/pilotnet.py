import torch
import torch.nn as nn




class PilotNet(nn.Module):
    def __init__(self, pretrained=False):
        super(PilotNet, self).__init__()
        default_input_channels = 3
        self.bn_1 = nn.BatchNorm2d(default_input_channels, eps=0.001)
        self.conv2d_1 = nn.Conv2d(default_input_channels, 24, 5, 2)
        self.conv2d_2 = nn.Conv2d(24, 36, 5, 2)
        self.conv2d_3 = nn.Conv2d(36, 48, 5, 2)
        self.conv2d_4 = nn.Conv2d(48, 64, 3, 1)
        self.conv2d_5 = nn.Conv2d(64, 64, 3, 1)

        self.classifier = nn.Sequential(
            nn.Linear(69696, 1164),
            # nn.Linear(1 * 18 * 64, 1164),
            nn.Linear(1164, 100),
            nn.Linear(100, 50),
            nn.Linear(50, 10),
            nn.Linear(10, 2)
        )



    def forward(self, x):
        x = self.bn_1(x)
        x = self.conv2d_1(x)
        x = torch.relu(x)
        x = self.conv2d_2(x)
        x = torch.relu(x)
        x = self.conv2d_3(x)
        x = torch.relu(x)
        x = self.conv2d_4(x)
        x = torch.relu(x)
        x = self.conv2d_5(x)
        x = torch.relu(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
