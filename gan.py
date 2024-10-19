import torch
from torch import nn

from modelblocks import LinearBlock


class Generator(nn.Module):
    # TODO: take in location labels, add one-hot month vector and then transform location
    #  via one linear layer before adding to another layer
    def __init__(self, c):
        super(Generator, self).__init__()
        self.num_layers = 12
        self.input_size = c["g_input_size"] * 2
        linear_params = {"num_layers": 3, "activation": "relu",
                         "input_size": 15, "layer_neurons": [15, 100, 100]}
        # self.use_bn = c["use_batch_norm"]    # determine to use batch normalization
        self.labelsLinearBlock = LinearBlock(linear_params)

        self.activation = nn.ReLU()
        input_len = self.input_size
        linear_out_size = input_len * 2
        self.map1_linear = nn.Linear(input_len, linear_out_size)
        input_len = linear_out_size
        linear_out_size = input_len * 2
        self.map2_linear = nn.Linear(input_len, linear_out_size)

        input_len = linear_out_size
        linear_out_size = input_len // 8
        self.map3_linear = nn.Linear(input_len, linear_out_size)

        # shape here should input_size * 1, do a dim expansion on the second dim here
        in_chan_4, out_chan_4 = 1, 10
        self.map4_deconv = nn.ConvTranspose1d(in_channels=in_chan_4,
                                              out_channels=out_chan_4,
                                              kernel_size=3)
        self.map4_bn = nn.BatchNorm1d(num_features=out_chan_4)
        out_chan_5 = 10
        self.map5_deconv = nn.ConvTranspose1d(in_channels=out_chan_4,
                                              out_channels=out_chan_5,
                                              kernel_size=3)
        self.map5_bn = nn.BatchNorm1d(num_features=out_chan_5)
        out_chan_6 = 64
        self.map6_deconv = nn.ConvTranspose1d(in_channels=out_chan_5,
                                              out_channels=out_chan_6,
                                              kernel_size=5)
        self.map6_bn = nn.BatchNorm1d(num_features=out_chan_6)

        out_chan_7 = 112
        self.map7_deconv = nn.ConvTranspose1d(in_channels=out_chan_6,
                                              out_channels=out_chan_7,
                                              kernel_size=5)

        # there should be an expand dim here
        out_chan_8 = 2
        self.avg_pool = nn.AvgPool2d(kernel_size=4)
        self.map7_conv = nn.Conv1d(in_channels=out_chan_7,
                                   out_channels=out_chan_7//4,
                                   kernel_size=1,
                                   stride=4) # changed from conv2d to 1d
        self.map7_bn = nn.BatchNorm1d(num_features=out_chan_7//4)
        self.map8_conv = nn.Conv2d(in_channels=1,
                                   out_channels=out_chan_8,
                                   kernel_size=(5, 5))
        self.map8_bn = nn.BatchNorm2d(num_features=out_chan_8)
        out_chan_9 = 4
        self.map9_conv = nn.Conv2d(in_channels=out_chan_8,
                                   out_channels=out_chan_9,
                                   kernel_size=(5, 5))
        self.map9_bn = nn.BatchNorm2d(num_features=out_chan_9)
        out_chan_10 = 16
        self.map10_conv = nn.Conv2d(in_channels=out_chan_9,
                                    out_channels=out_chan_10,
                                    kernel_size=(5, 5))
        self.map10_bn = nn.BatchNorm2d(num_features=out_chan_10)
        out_chan_11 = 24
        self.map11_conv = nn.Conv2d(in_channels=out_chan_10,
                                    out_channels=out_chan_11,
                                    kernel_size=(5, 5))
        self.map11_bn = nn.BatchNorm2d(num_features=out_chan_11)

        self.map12_conv = nn.Conv2d(in_channels=out_chan_11,
                                    out_channels=out_chan_11,
                                    kernel_size=(5, 5))

    def forward(self, x):
        # x_input = x
        x1 = x[:, 0:100]
        x2 = x[:, 100:]
        x2 = self.labelsLinearBlock(x2)
        x = torch.cat((x1, x2), dim=1)  # concatenate vectors into 200 X 1 vector
        # transform the [one-hot, loc_x, loc_y] into a 100 dim z to add to z-noise
        x = self.map1_linear(x)
        x = self.activation(x)

        x = self.map2_linear(x)
        x = self.activation(x)

        x = self.map3_linear(x)
        x = self.activation(x)

        x = x.unsqueeze(dim=1)
        x = self.map4_deconv(x)
        x = self.map4_bn(x)
        x = self.activation(x)

        x = self.map5_deconv(x)
        x = self.map5_bn(x)
        x = self.activation(x)

        x = self.map6_deconv(x)
        x = self.map6_bn(x)
        x = self.activation(x)

        x = self.map7_deconv(x)
        x = self.activation(x)

        x = self.map7_conv(x)
        # x = self.avg_pool(x)
        x = self.map7_bn(x)
        x = self.activation(x)

        x = x.unsqueeze(dim=1)
        x = self.map8_conv(x)
        x = self.map8_bn(x)
        x = self.activation(x)

        x = self.map9_conv(x)
        x = self.map9_bn(x)
        x = nn.Tanh()(x)

        x = self.map10_conv(x)
        x = self.map10_bn(x)
        x = nn.Tanh()(x)

        x = self.map11_conv(x)
        x = self.map11_bn(x)
        x = nn.Tanh()(x)
        # changed last 3 layers to Tanh. Kept batchNorm

        x = self.map12_conv(x)

        return x


class GeneratorGMST(nn.Module):
    """
    Uses GMST as input for tracking the average surface temperature of the entire globe.
    """
    def __init__(self, c):
        super(GeneratorGMST, self).__init__()
        self.num_layers = 12
        self.input_size = c["g_input_size"] * 2     # Test to see if embedding dim has impact on model performance.
        linear_params = {"num_layers": 3, "activation": "relu",
                         "input_size": 19, "layer_neurons": [19, 100, 100]}
        # self.use_bn = c["use_batch_norm"]    # determine to use batch normalization
        self.labelsLinearBlock = LinearBlock(linear_params)

        self.activation = nn.ReLU()
        input_len = self.input_size
        linear_out_size = input_len * 2
        self.map1_linear = nn.Linear(input_len, linear_out_size)
        input_len = linear_out_size
        linear_out_size = input_len * 2
        self.map2_linear = nn.Linear(input_len, linear_out_size)

        input_len = linear_out_size
        linear_out_size = input_len // 8
        self.map3_linear = nn.Linear(input_len, linear_out_size)

        # shape here should input_size * 1, do a dim expansion on the second dim here
        in_chan_4, out_chan_4 = 1, 10
        self.map4_deconv = nn.ConvTranspose1d(in_channels=in_chan_4,
                                              out_channels=out_chan_4,
                                              kernel_size=3)
        self.map4_bn = nn.BatchNorm1d(num_features=out_chan_4)
        out_chan_5 = 10
        self.map5_deconv = nn.ConvTranspose1d(in_channels=out_chan_4,
                                              out_channels=out_chan_5,
                                              kernel_size=3)
        self.map5_bn = nn.BatchNorm1d(num_features=out_chan_5)
        out_chan_6 = 64
        self.map6_deconv = nn.ConvTranspose1d(in_channels=out_chan_5,
                                              out_channels=out_chan_6,
                                              kernel_size=5)
        self.map6_bn = nn.BatchNorm1d(num_features=out_chan_6)

        out_chan_7 = 112
        self.map7_deconv = nn.ConvTranspose1d(in_channels=out_chan_6,
                                              out_channels=out_chan_7,
                                              kernel_size=5)

        # there should be an expand dim here
        out_chan_8 = 2
        self.avg_pool = nn.AvgPool2d(kernel_size=4)
        self.map7_conv = nn.Conv1d(in_channels=out_chan_7,
                                   out_channels=out_chan_7//4,
                                   kernel_size=1,
                                   stride=4) # changed from conv2d to 1d
        self.map7_bn = nn.BatchNorm1d(num_features=out_chan_7//4)
        self.map8_conv = nn.Conv2d(in_channels=1,
                                   out_channels=out_chan_8,
                                   kernel_size=(5, 5))
        self.map8_bn = nn.BatchNorm2d(num_features=out_chan_8)
        out_chan_9 = 4
        self.map9_conv = nn.Conv2d(in_channels=out_chan_8,
                                   out_channels=out_chan_9,
                                   kernel_size=(5, 5))
        self.map9_bn = nn.BatchNorm2d(num_features=out_chan_9)
        out_chan_10 = 16
        self.map10_conv = nn.Conv2d(in_channels=out_chan_9,
                                    out_channels=out_chan_10,
                                    kernel_size=(5, 5))
        self.map10_bn = nn.BatchNorm2d(num_features=out_chan_10)
        out_chan_11 = 24
        self.map11_conv = nn.Conv2d(in_channels=out_chan_10,
                                    out_channels=out_chan_11,
                                    kernel_size=(5, 5))
        self.map11_bn = nn.BatchNorm2d(num_features=out_chan_11)

        self.map12_conv = nn.Conv2d(in_channels=out_chan_11,
                                    out_channels=out_chan_11,
                                    kernel_size=(5, 5))

    def forward(self, x):
        # x_input = x
        x1 = x[:, 0:100]
        x2 = x[:, 100:]
        x2 = self.labelsLinearBlock(x2)
        x = torch.cat((x1, x2), dim=1)  # concatenate vectors into 200 X 1 vector
        # transform the [one-hot, loc_x, loc_y] into a 100 dim z to add to z-noise
        x = self.map1_linear(x)
        x = self.activation(x)

        x = self.map2_linear(x)
        x = self.activation(x)

        x = self.map3_linear(x)
        x = self.activation(x)

        x = x.unsqueeze(dim=1)
        x = self.map4_deconv(x)
        x = self.map4_bn(x)
        x = self.activation(x)

        x = self.map5_deconv(x)
        x = self.map5_bn(x)
        x = self.activation(x)

        x = self.map6_deconv(x)
        x = self.map6_bn(x)
        x = self.activation(x)

        x = self.map7_deconv(x)
        x = self.activation(x)

        x = self.map7_conv(x)
        # x = self.avg_pool(x)
        x = self.map7_bn(x)
        x = self.activation(x)

        x = x.unsqueeze(dim=1)
        x = self.map8_conv(x)
        x = self.map8_bn(x)
        x = self.activation(x)

        x = self.map9_conv(x)
        x = self.map9_bn(x)
        x = nn.Tanh()(x)

        x = self.map10_conv(x)
        x = self.map10_bn(x)
        x = nn.Tanh()(x)

        x = self.map11_conv(x)
        x = self.map11_bn(x)
        x = nn.Tanh()(x)
        # changed last 3 layers to Tanh. Kept batchNorm

        x = self.map12_conv(x)

        return x


class DiscriminatorGMST(nn.Module):
    def __init__(self):
        super(DiscriminatorGMST, self).__init__()
        linear_params = {"num_layers": 3, "activation": "relu",
                         "input_size": 19, "layer_neurons": [19, 50, 100]}
        self.label_map1 = LinearBlock(linear_params)
        in_chan_1 = 24
        out_chan_1 = in_chan_1
        self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.map1_conv = nn.Conv2d(in_channels=in_chan_1,
                                   out_channels=out_chan_1,
                                   kernel_size=3,
                                   padding='same',
                                   padding_mode='replicate')
        out_chan2 = 24
        self.map2_conv = nn.Conv2d(in_channels=out_chan_1,
                                   out_channels=out_chan2,
                                   kernel_size=3,
                                   padding='same',
                                   padding_mode='replicate')
        out_chan3 = out_chan2
        self.map3_conv = nn.Conv2d(in_channels=out_chan2,
                                   out_channels=out_chan3,
                                   kernel_size=3)
        out_chan4 = out_chan3
        self.map4_conv = nn.Conv2d(in_channels=out_chan3,
                                   out_channels=out_chan4,
                                   kernel_size=3)

        out_chan5 = out_chan4
        self.map5_conv = nn.Conv2d(in_channels=out_chan4,
                                   out_channels=out_chan5,
                                   kernel_size=3)
        out_chan6 = out_chan5
        self.map6_conv = nn.Conv2d(in_channels=out_chan5,
                                   out_channels=out_chan6,
                                   kernel_size=2)

        self.map7_linear = nn.Linear(in_features=24, out_features=100) # concatenate after this layer
        self.map8_linear = nn.Linear(in_features=200, out_features=150)
        self.map9_linear = nn.Linear(in_features=150, out_features=50)
        self.map10_linear = nn.Linear(in_features=50, out_features=1)

    def forward(self, x, y):
        y = self.label_map1(y)
        x = self.map1_conv(x)
        x = self.activation(x)

        x = self.map2_conv(x)
        x = self.activation(x)

        x = self.map3_conv(x)
        x = self.activation(x)

        x = self.map4_conv(x)
        x = self.activation(x)

        x = self.map5_conv(x)
        x = self.activation(x)

        x = self.map6_conv(x)
        x = self.activation(x)

        x = torch.flatten(x, start_dim=1)     # remove the #1 idx dimension
        x = self.map7_linear(x)
        x = torch.cat((x, y), dim=1)    # add label information to this layer
        x = self.activation(x)

        x = self.map8_linear(x)
        x = self.activation(x)

        x = self.map9_linear(x)
        x = self.activation(x)

        x = self.map10_linear(x)
        # TEST SIGMOID HERE
        # x = nn.Sigmoid()(x)
        # x = self.activation(x)

        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        linear_params = {"num_layers": 3, "activation": "relu",
                         "input_size": 15, "layer_neurons": [15, 50, 100]}
        self.label_map1 = LinearBlock(linear_params)
        in_chan_1 = 24
        out_chan_1 = in_chan_1
        self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.map1_conv = nn.Conv2d(in_channels=in_chan_1,
                                   out_channels=out_chan_1,
                                   kernel_size=3,
                                   padding='same',
                                   padding_mode='replicate')
        out_chan2 = 24
        self.map2_conv = nn.Conv2d(in_channels=out_chan_1,
                                   out_channels=out_chan2,
                                   kernel_size=3,
                                   padding='same',
                                   padding_mode='replicate')
        out_chan3 = out_chan2
        self.map3_conv = nn.Conv2d(in_channels=out_chan2,
                                   out_channels=out_chan3,
                                   kernel_size=3)
        out_chan4 = out_chan3
        self.map4_conv = nn.Conv2d(in_channels=out_chan3,
                                   out_channels=out_chan4,
                                   kernel_size=3)

        out_chan5 = out_chan4
        self.map5_conv = nn.Conv2d(in_channels=out_chan4,
                                   out_channels=out_chan5,
                                   kernel_size=3)
        out_chan6 = out_chan5
        self.map6_conv = nn.Conv2d(in_channels=out_chan5,
                                   out_channels=out_chan6,
                                   kernel_size=2)

        self.map7_linear = nn.Linear(in_features=24, out_features=100) # concatenate after this layer
        self.map8_linear = nn.Linear(in_features=200, out_features=150)
        self.map9_linear = nn.Linear(in_features=150, out_features=50)
        self.map10_linear = nn.Linear(in_features=50, out_features=1)

    def forward(self, x, y):
        y = self.label_map1(y)
        x = self.map1_conv(x)
        x = self.activation(x)

        x = self.map2_conv(x)
        x = self.activation(x)

        x = self.map3_conv(x)
        x = self.activation(x)

        x = self.map4_conv(x)
        x = self.activation(x)

        x = self.map5_conv(x)
        x = self.activation(x)

        x = self.map6_conv(x)
        x = self.activation(x)

        x = torch.flatten(x, start_dim=1)     # remove the #1 idx dimension
        x = self.map7_linear(x)
        x = torch.cat((x, y), dim=1)    # add label information to this layer
        x = self.activation(x)

        x = self.map8_linear(x)
        x = self.activation(x)

        x = self.map9_linear(x)
        x = self.activation(x)

        x = self.map10_linear(x)
        # TEST SIGMOID HERE
        # x = nn.Sigmoid()(x)
        # x = self.activation(x)

        return x


class DiscriminatorCyclic(nn.Module):
    def __init__(self):
        super(DiscriminatorCyclic, self).__init__()
        linear_params = {"num_layers": 3, "activation": "relu",
                         "input_size": 5, "layer_neurons": [5, 64, 100]}
        self.label_map1 = LinearBlock(linear_params)
        in_chan_1 = 24
        out_chan_1 = in_chan_1
        self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.map1_conv = nn.Conv2d(in_channels=in_chan_1,
                                   out_channels=out_chan_1,
                                   kernel_size=3,
                                   padding='same',
                                   padding_mode='replicate')
        out_chan2 = 24
        self.map2_conv = nn.Conv2d(in_channels=out_chan_1,
                                   out_channels=out_chan2,
                                   kernel_size=3,
                                   padding='same',
                                   padding_mode='replicate')
        out_chan3 = out_chan2
        self.map3_conv = nn.Conv2d(in_channels=out_chan2,
                                   out_channels=out_chan3,
                                   kernel_size=3)
        out_chan4 = out_chan3
        self.map4_conv = nn.Conv2d(in_channels=out_chan3,
                                   out_channels=out_chan4,
                                   kernel_size=3)

        out_chan5 = out_chan4
        self.map5_conv = nn.Conv2d(in_channels=out_chan4,
                                   out_channels=out_chan5,
                                   kernel_size=3)
        out_chan6 = out_chan5
        self.map6_conv = nn.Conv2d(in_channels=out_chan5,
                                   out_channels=out_chan6,
                                   kernel_size=2)

        self.map7_linear = nn.Linear(in_features=24, out_features=100) # concatenate after this layer
        self.map8_linear = nn.Linear(in_features=200, out_features=150)
        self.map9_linear = nn.Linear(in_features=150, out_features=50)
        self.map10_linear = nn.Linear(in_features=50, out_features=1)

    def forward(self, x, y):
        y = self.label_map1(y)
        x = self.map1_conv(x)
        x = self.activation(x)

        x = self.map2_conv(x)
        x = self.activation(x)

        x = self.map3_conv(x)
        x = self.activation(x)

        x = self.map4_conv(x)
        x = self.activation(x)

        x = self.map5_conv(x)
        x = self.activation(x)

        x = self.map6_conv(x)
        x = self.activation(x)

        x = torch.flatten(x, start_dim=1)     # remove the #1 idx dimension
        x = self.map7_linear(x)
        x = torch.cat((x, y), dim=1)    # add label information to this layer
        x = self.activation(x)

        x = self.map8_linear(x)
        x = self.activation(x)

        x = self.map9_linear(x)
        x = self.activation(x)

        x = self.map10_linear(x)
        # TEST SIGMOID HERE
        # x = nn.Sigmoid()(x)
        # x = self.activation(x)

        return x


class GeneratorCyclic(nn.Module):
    """This uses the new cyclic labels for modelling the temperatures"""
    def __init__(self, c):
        super(GeneratorCyclic, self).__init__()
        self.num_layers = 12
        self.input_size = c["g_input_size"] * 2
        linear_params = {"num_layers": 3, "activation": "tanh",
                         "input_size": 5, "layer_neurons": [5, 64, 100]}
        # self.use_bn = c["use_batch_norm"]    # determine to use batch normalization
        self.labelsLinearBlock = LinearBlock(linear_params)

        self.activation = nn.ReLU()
        input_len = self.input_size
        linear_out_size = input_len * 2
        self.map1_linear = nn.Linear(input_len, linear_out_size)
        input_len = linear_out_size
        linear_out_size = input_len * 2
        self.map2_linear = nn.Linear(input_len, linear_out_size)

        input_len = linear_out_size
        linear_out_size = input_len // 8
        self.map3_linear = nn.Linear(input_len, linear_out_size)

        # shape here should input_size * 1, do a dim expansion on the second dim here
        in_chan_4, out_chan_4 = 1, 10
        self.map4_deconv = nn.ConvTranspose1d(in_channels=in_chan_4,
                                              out_channels=out_chan_4,
                                              kernel_size=3)
        self.map4_bn = nn.BatchNorm1d(num_features=out_chan_4)
        out_chan_5 = 10
        self.map5_deconv = nn.ConvTranspose1d(in_channels=out_chan_4,
                                              out_channels=out_chan_5,
                                              kernel_size=3)
        self.map5_bn = nn.BatchNorm1d(num_features=out_chan_5)
        out_chan_6 = 64
        self.map6_deconv = nn.ConvTranspose1d(in_channels=out_chan_5,
                                              out_channels=out_chan_6,
                                              kernel_size=5)
        self.map6_bn = nn.BatchNorm1d(num_features=out_chan_6)

        out_chan_7 = 112
        self.map7_deconv = nn.ConvTranspose1d(in_channels=out_chan_6,
                                              out_channels=out_chan_7,
                                              kernel_size=5)

        # there should be an expand dim here
        out_chan_8 = 2
        self.avg_pool = nn.AvgPool2d(kernel_size=4)
        self.map7_conv = nn.Conv1d(in_channels=out_chan_7,
                                   out_channels=out_chan_7//4,
                                   kernel_size=1,
                                   stride=4) # changed from conv2d to 1d
        self.map7_bn = nn.BatchNorm1d(num_features=out_chan_7//4)
        self.map8_conv = nn.Conv2d(in_channels=1,
                                   out_channels=out_chan_8,
                                   kernel_size=(5, 5))
        self.map8_bn = nn.BatchNorm2d(num_features=out_chan_8)
        out_chan_9 = 4
        self.map9_conv = nn.Conv2d(in_channels=out_chan_8,
                                   out_channels=out_chan_9,
                                   kernel_size=(5, 5))
        self.map9_bn = nn.BatchNorm2d(num_features=out_chan_9)
        out_chan_10 = 16
        self.map10_conv = nn.Conv2d(in_channels=out_chan_9,
                                    out_channels=out_chan_10,
                                    kernel_size=(5, 5))
        self.map10_bn = nn.BatchNorm2d(num_features=out_chan_10)
        out_chan_11 = 24
        self.map11_conv = nn.Conv2d(in_channels=out_chan_10,
                                    out_channels=out_chan_11,
                                    kernel_size=(5, 5))
        self.map11_bn = nn.BatchNorm2d(num_features=out_chan_11)

        self.map12_conv = nn.Conv2d(in_channels=out_chan_11,
                                    out_channels=out_chan_11,
                                    kernel_size=(5, 5))

    def forward(self, x):
        # x_input = x
        x1 = x[:, 0:100]
        x2 = x[:, 100:]
        x2 = self.labelsLinearBlock(x2)
        x = torch.cat((x1, x2), dim=1)  # concatenate vectors into 200 X 1 vector
        # transform the [one-hot, loc_x, loc_y] into a 100 dim z to add to z-noise
        x = self.map1_linear(x)
        x = self.activation(x)

        x = self.map2_linear(x)
        x = self.activation(x)

        x = self.map3_linear(x)
        x = self.activation(x)

        x = x.unsqueeze(dim=1)
        x = self.map4_deconv(x)
        x = self.map4_bn(x)
        x = self.activation(x)

        x = self.map5_deconv(x)
        x = self.map5_bn(x)
        x = self.activation(x)

        x = self.map6_deconv(x)
        x = self.map6_bn(x)
        x = self.activation(x)

        x = self.map7_deconv(x)
        x = self.activation(x)

        x = self.map7_conv(x)
        # x = self.avg_pool(x)
        x = self.map7_bn(x)
        x = self.activation(x)

        x = x.unsqueeze(dim=1)
        x = self.map8_conv(x)
        x = self.map8_bn(x)
        x = self.activation(x)

        x = self.map9_conv(x)
        x = self.map9_bn(x)
        x = nn.Tanh()(x)

        x = self.map10_conv(x)
        x = self.map10_bn(x)
        x = nn.Tanh()(x)

        x = self.map11_conv(x)
        x = self.map11_bn(x)
        x = nn.Tanh()(x)
        # changed last 3 layers to Tanh. Kept batchNorm

        x = self.map12_conv(x)

        return x


class DiscriminatorS(nn.Module):
    """
    This module uses the maximum variance slice of the image as input to the discriminator.

    """
    def __init__(self):
        super(DiscriminatorS, self).__init__()
        linear_params = {"num_layers": 3, "activation": "relu",
                         "input_size": 15, "layer_neurons": [15, 50, 100]}
        self.label_map1 = LinearBlock(linear_params)
        in_chan_1 = 1   # this only takes the image slice with maximum variance
        out_chan_1 = in_chan_1
        self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.map1_conv = nn.Conv2d(in_channels=in_chan_1,
                                   out_channels=out_chan_1,
                                   kernel_size=3,
                                   padding='same',
                                   padding_mode='replicate')
        out_chan2 = 2
        self.map2_conv = nn.Conv2d(in_channels=out_chan_1,
                                   out_channels=out_chan2,
                                   kernel_size=3,
                                   padding='same',
                                   padding_mode='replicate')
        out_chan3 = 4
        self.map3_conv = nn.Conv2d(in_channels=out_chan2,
                                   out_channels=out_chan3,
                                   kernel_size=3)
        out_chan4 = 16
        self.map4_conv = nn.Conv2d(in_channels=out_chan3,
                                   out_channels=out_chan4,
                                   kernel_size=3)

        out_chan5 = 16
        self.map5_conv = nn.Conv2d(in_channels=out_chan4,
                                   out_channels=out_chan5,
                                   kernel_size=3)
        out_chan6 = out_chan5
        self.map6_conv = nn.Conv2d(in_channels=out_chan5,
                                   out_channels=out_chan6,
                                   kernel_size=2)

        self.map7_linear = nn.Linear(in_features=16, out_features=16)   # concatenate after this layer
        self.map8_linear = nn.Linear(in_features=116, out_features=128)
        self.map9_linear = nn.Linear(in_features=128, out_features=64)
        self.map10_linear = nn.Linear(in_features=64, out_features=1)

    def forward(self, x, y):
        y = self.label_map1(y)

        # get desired slice

        x = x.unsqueeze(1)  # this is to get one channel
        x = self.map1_conv(x)
        x = self.activation(x)

        x = self.map2_conv(x)
        x = self.activation(x)

        x = self.map3_conv(x)
        x = self.activation(x)

        x = self.map4_conv(x)
        x = self.activation(x)

        x = self.map5_conv(x)
        x = self.activation(x)

        x = self.map6_conv(x)
        x = self.activation(x)

        x = torch.flatten(x, start_dim=1)     # remove the #1 idx dimension
        x = self.map7_linear(x)
        x = torch.cat((x, y), dim=1)    # add label information to this layer
        x = self.activation(x)

        x = self.map8_linear(x)
        x = self.activation(x)

        x = self.map9_linear(x)
        x = self.activation(x)

        x = self.map10_linear(x)
        # TEST SIGMOID HERE
        # x = nn.Sigmoid()(x)
        # x = self.activation(x)

        return x


class DiscriminatorT(nn.Module):
    """Temporal discriminator is evaluated on the GRADIENTS of its images"""

    def __init__(self, max_pool=False):
        """Vanilla Temporal Discriminator via 3D convolutions"""
        super(DiscriminatorT, self).__init__()
        linear_params_label = {
            "num_layers": 3,
            "activation": "relu",
            "input_size": 15,
            "layer_neurons": [15, 64, 100]
        }
        LabelMap = LinearBlock(linear_params_label)

        linear_params = {"num_layers": 4,
                         "activation": "relu",
                         "input_size": 164,
                         "layer_neurons": [164, 200, 100, 1]}

        self.map2_conv2D = nn.Conv2d(in_channels=23, out_channels=16, kernel_size=(3, 3))
        self.map3_conv2D = nn.Conv2d(in_channels=16, out_channels=4, kernel_size=(3, 3))
        self.map4_linear = LinearBlock(linear_params)
        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.label_map = LabelMap

    def forward(self, x, y):
        a_leaky_relu = 0.2  # as in DCGAN
        y = self.label_map(y)  # map y into a higher dim space for representation (should test without mapping later)

        # check if input shape is correct
        assert list(x.shape)[1:] == [24, 8, 8]  # channels X Hin X Win
        # expand dim for convnet
        x = x[:, 0:23, :, :] - x[:, 1:24, :, :]  # take the hourly gradient
        x = self.map2_conv2D(x)
        x = self.activation(x)

        x = self.map3_conv2D(x)
        x = self.activation(x)

        x = torch.flatten(x, start_dim=1)
        x = torch.cat((x, y), dim=1)  # add label information to this layer
        x = self.map4_linear(x)
        return x


class DiscriminatorTGMST(nn.Module):
    """Temporal discriminator is evaluated on the GRADIENTS of its images"""

    def __init__(self, max_pool=False):
        """Vanilla Temporal Discriminator via 3D convolutions"""
        super(DiscriminatorTGMST, self).__init__()
        linear_params_label = {
            "num_layers": 3,
            "activation": "relu",
            "input_size": 19,
            "layer_neurons": [19, 64, 100]
        }   # This is for the learned label embedding network. 19 is the size of the label vector
        LabelMap = LinearBlock(linear_params_label)

        linear_params = {"num_layers": 4,
                         "activation": "relu",
                         "input_size": 164,
                         "layer_neurons": [164, 200, 100, 1]}   # This is for the linear block in the later layers.

        self.map2_conv2D = nn.Conv2d(in_channels=23, out_channels=16, kernel_size=(3, 3))
        self.map3_conv2D = nn.Conv2d(in_channels=16, out_channels=4, kernel_size=(3, 3))
        self.map4_linear = LinearBlock(linear_params)
        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.label_map = LabelMap

    def forward(self, x, y):
        a_leaky_relu = 0.2  # as in DCGAN
        y = self.label_map(y)  # map y into a higher dim space for representation (should test without mapping later)

        # check if input shape is correct
        assert list(x.shape)[1:] == [24, 8, 8]  # channels X Hin X Win
        # expand dim for convnet
        x = x[:, 0:23, :, :] - x[:, 1:24, :, :]  # Hourly gradient computation.
        x = self.map2_conv2D(x)
        x = self.activation(x)

        x = self.map3_conv2D(x)
        x = self.activation(x)

        x = torch.flatten(x, start_dim=1)
        x = torch.cat((x, y), dim=1)  # Label information is included to this layer.
        x = self.map4_linear(x)
        return x


class DiscriminatorTDeep(nn.Module):
    """
    Temporal discriminator is evaluated on the GRADIENTS of its images...
    This is deeper than the one in the paper
    """

    def __init__(self, max_pool=False):
        """Vanilla Temporal Discriminator via 3D convolutions"""
        super(DiscriminatorTDeep, self).__init__()
        linear_params = {"num_layers": 4, "activation": "relu", "input_size": 164, "layer_neurons": [164, 128, 64, 1]}
        linear_params_label = {"num_layers": 3, "activation": "relu", "input_size": 15, "layer_neurons": [15, 50, 100]}

        self.map2_conv2D = nn.Conv2d(in_channels=23, out_channels=16, kernel_size=(3, 3),
                                     padding='same', padding_mode='replicate')
        self.map3_conv2D = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3),
                                     padding='same', padding_mode='replicate')
        self.map4_conv2D = nn.Conv2d(in_channels=16, out_channels=4, kernel_size=(3, 3))
        self.map5_conv2D = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(3, 3))
        self.map6_linear = LinearBlock(linear_params)
        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.label_map = LinearBlock(linear_params_label)

    def forward(self, x, y):
        a_leaky_relu = 0.2  # as in DCGAN
        y = self.label_map(y)  # map y into a higher dim space for representation (should test without mapping later)

        # check if input shape is correct
        assert list(x.shape)[1:] == [24, 8, 8]  # channels X Hin X Win
        # expand dim for convnet
        x = x[:, 0:23, :, :] - x[:, 1:24, :, :]  # take the hourly gradient
        x = self.map2_conv2D(x)
        x = self.activation(x)

        x = self.map3_conv2D(x)
        x = self.activation(x)

        x = self.map4_conv2D(x)
        x = self.activation(x)

        x = self.map5_conv2D(x)
        x = self.activation(x)

        x = torch.flatten(x, start_dim=1)
        x = torch.cat((x, y), dim=1)  # add label information to this layer
        x = self.map6_linear(x)
        return x
