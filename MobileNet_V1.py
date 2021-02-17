import torch.nn as nn
from torchsummary import summary

########################################################################################################################
################################# unofficial Pytorch implementation of MobileNet Model################################
##################################### Source : https://arxiv.org/pdf/1704.04861.pdf ################################


class DW_Conv_Block(nn.Module):
    def __init__(self,in_channels,out_channels,stride):
        super(DW_Conv_Block, self).__init__()
        self.stride = stride
        #According to the Table 1 in the paper, The last DW_Module should get input with the size of (1024,7,7) and
        # returns output with the same size, For this purpose, the padding should be equal to 4. In other layers, padding =1
        if in_channels == 1024:
            pw_padding = 4
        else:
            pw_padding = 1
        self.dw_conv = nn.Conv2d(in_channels=in_channels,
                  out_channels=in_channels,
                  kernel_size=3,
                  stride=self.stride,
                  padding=pw_padding,
                  groups=in_channels)
        self.bn1 = nn.BatchNorm2d(in_channels)

        self.pw_conv = nn.Conv2d(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=1,
                                 padding=0,
                                 stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        x = self.dw_conv(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pw_conv(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class MobileNetV1(nn.Module):
    def __init__(self,in_channels = 3,num_classes=1000,width_multiplier = 1.0):
        """

        :param in_channels: Number of Input channels for the whole Model, default:3(RGB images)
        :param num_classes: Number of classes for the whole Model, default:1000(ImageNet Dataset)
        :param width_multiplier: ( Called as α ∈ (0,1], and default value of 1, the smaller the α is, the thinner the model.
        # it multiplies to all the number of input_channels, and output_channels in the whole network, except
        1: input channels of input data( image),
        2- output_channels of the last layer( Number of classes)
        """

        #(# of Dw_Blocks, Output of the block, Stride)
        self.config_list = \
            [(1, 64, 1),
             (1, 128, 2),
             (1, 128, 1),
             (1, 256, 2),
             (1, 256, 1),
             (1, 512, 2),
             (5, 512, 1),
             (1, 1024, 2),
             (1, 1024, 2)]

        self.width_multiplier = width_multiplier

        super(MobileNetV1, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=int(32 * self.width_multiplier),
                              kernel_size=3,
                              stride=2,
                              padding=1)
        self.bn = nn.BatchNorm2d(int(32 * self.width_multiplier))
        self.relu = nn.ReLU(inplace=True)

        # create all the DepthWise block layer via nn.Sequential
        self.dw_pw_layers = self._make_layers()
        #if (7 * self.width_multiplier) < 1.0:
           # pooling_stride = 1
        #else:
           # pooling_stride = int(7 * self.width_multiplier)

        #self.avgpool = nn.AvgPool2d(pooling_stride)
        self.fc = nn.Linear(int(1024 * self.width_multiplier),num_classes)


    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dw_pw_layers(x)
        #x = self.avgpool(x)
        x = nn.AvgPool2d(x.size(2))(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x



    def _make_layers(self):

        layers = []
        previous_out_channel = 0

        for pair_numbers,pair_output_channels,stride_of_depth_wise in self.config_list:

            for number_of_pairs in range(pair_numbers):

                if len(layers) == 0 :
                    # if this is the firs DW_Modulde, then the input_channels = 32
                    input_channels = int (32 * self.width_multiplier)
                else:
                    #else the input_channels should be equal to the otput channels of the previous DW-Module
                    input_channels = previous_out_channel

                layers.append(DW_Conv_Block(out_channels=int(pair_output_channels * self.width_multiplier),
                                                    stride=stride_of_depth_wise,
                                                    in_channels=input_channels))
                previous_out_channel = int( pair_output_channels * self.width_multiplier)

        return nn.Sequential(*layers)




mobile = MobileNetV1(in_channels=3, num_classes=1000,width_multiplier=1.0)
summary(mobile, input_size=(3, 224, 224), device='cpu')