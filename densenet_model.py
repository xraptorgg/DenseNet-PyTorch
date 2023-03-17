"""
Implementation of original DenseNetBC architecture in PyTorch, from the paper
"Densely Connected Convolutional Networks" by Gao Huang et al.
at: https://arxiv.org/pdf/1608.06993.pdf
"""


# importing necessary libraries

import torch
import torch.nn as nn

# ResNet architecture
# supports DenseNet-121, 169, 201, 264
# need to pass in the depth as argument



class DenseNet(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()

        self.block_config = {
            121 : [6, 12, 24, 16],
            169 : [6, 12, 32, 32],
            201 : [6, 12, 48, 32],
            264 : [6, 12, 64, 48]
        }

        self.k = 32

        self.bn1 = nn.BatchNorm2d(3)
        self.relu = nn.ReLU()
        self.conv_1 = nn.Conv2d(in_channels = 3, out_channels = 2 * self.k, kernel_size = 7, stride = 2, padding = 3)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        self.dense_blocks = nn.ModuleList([])
        self.k_0 = 2 * self.k

        for i, l in enumerate(self.block_config[config]):
            self.dense_blocks.append(dense_block(num_layers = l, in_channels = self.k_0, growth_rate = self.k))
            self.k_0 += self.k * l

            if i != len(self.block_config[config]) - 1:
                self.dense_blocks.append(transition_layer(in_channels = self.k_0, theta = 0.5))
                self.k_0 //= 2

        self.flatten = nn.Flatten()
        self.bn2 = nn.BatchNorm2d(self.k_0)
        self.global_avgpool = nn.AvgPool2d(kernel_size = 7)
        self.fc = nn.Linear(in_features = self.k_0, out_features = num_classes)


        self.init_weights()


    def init_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
                

    def forward(self, x):
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv_1(x)
        x = self.maxpool(x)

        for layer in self.dense_blocks:
            x = layer(x)

        x = self.bn2(x)
        x = self.relu(x)
        x = self.global_avgpool(x)
        x = self.flatten(x)
        return self.fc(x)



class dense_block(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate):
        super().__init__()

        self.block = nn.ModuleList([])

        for i in range(num_layers):
            self.block.append(dense_layer(in_channels = in_channels + (i * growth_rate), growth_rate = growth_rate))


    def forward(self, x):
        for layer in self.block:
            x = layer(x)
        return x



class transition_layer(nn.Module):
    def __init__(self, in_channels, theta):
        super().__init__()

        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.conv_1x1 = nn.Conv2d(in_channels = in_channels, out_channels = int(theta * in_channels), kernel_size = 1, stride = 1, padding = 0)
        self.avgpool = nn.AvgPool2d(kernel_size = 2, stride = 2)

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv_1x1(x)
        return self.avgpool(x)



class dense_layer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.conv_1x1 = nn.Conv2d(in_channels = in_channels, out_channels = 4 * growth_rate, kernel_size = 1, stride = 1, padding = 0)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv_3x3 = nn.Conv2d(in_channels = 4 * growth_rate, out_channels = growth_rate, kernel_size = 3, stride = 1, padding = 1)

    def forward(self, x):
        feature_maps = x

        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv_1x1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv_3x3(x)

        return torch.cat([feature_maps, x], 1)