import torch.nn as nn
# import torch
import torch.nn.functional as F


class LinearBlock(nn.Module):
    """creates a multi-layer linear network.
    Might be a bit less efficient but is probably negligible"""

    def __init__(self, input_params):
        super(LinearBlock, self).__init__()
        self.num_layers = 0
        self.activation = input_params["activation"]
        input_size = input_params["input_size"]
        self.layers = nn.ModuleList()
        num_layers = input_params["num_layers"]
        layer_neurons = input_params["layer_neurons"]
        for layer in range(num_layers):
            neuron_count = layer_neurons[layer]
            # attr_name = "map{}_linear".format(layer)
            self.layers.append(nn.Linear(input_size, neuron_count))
            # self.layers.append(nn.ReLU())
            input_size = neuron_count
            self.num_layers += 1
        if self.activation == "Leaky_relu":
            self.activation = F.leaky_relu
        elif self.activation == "tanh":
            self.activation = F.tanh
        elif self.activation == "relu":
            self.activation = F.relu

    def forward(self, x):
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx == len(self.layers) - 1:
                return x    # do not apply activation on final output
            x = F.relu(x)
        return x

class DeConvBlock1D(nn.Module):
    def __init__(self, params):
        super(DeConvBlock1D, self).__init__()
        self.num_layers = 0
        self.activation = params["activation"]
        num_layers = params["num_layers"]
        input_channel = params["input_channels"]
        layer_channels = params["channel_counts"]
        layer_kernels = params["kernels"]
        paddings = params["paddings"]
        strides = params["strides"]
        for layer in range(num_layers):
            kernel = layer_kernels[layer]
            channels = layer_channels[layer]
            pad = paddings[layer]
            stride = strides[layer]
            attr_name = "map{}_deconv1d".format(layer)
            self.layers.append(attr_name)
            setattr(self, attr_name, nn.ConvTranspose1d(kernel_size=kernel, in_channels=input_channel,
                                                        out_channels=channels, padding=pad, stride=stride))
            input_channel = channels
            self.num_layers += 1

class ConvBlock1D(nn.Module):
    def __init__(self, params):
        super(ConvBlock1D, self).__init__()
        self.num_layers = 0
        self.activation = params["activation"]
        num_layers = params["num_layers"]
        input_channel = params["input_channels"]
        layer_channels = params["channel_counts"]
        layer_kernels = params["kernels"]
        paddings = params["paddings"]
        strides = params["strides"]
        for layer in range(num_layers):
            kernel = layer_kernels[layer]
            channels = layer_channels[layer]
            pad = paddings[layer]
            stride = strides[layer]
            attr_name = "map{}_conv1d".format(layer)
            self.layers.append(attr_name)
            setattr(self, attr_name, nn.Conv1d(kernel_size=kernel, in_channels=input_channel,
                                               out_channels=channels, padding=pad, stride=stride))
            input_channel = channels
            self.num_layers += 1

class ImageGenBlockTGAN(nn.Module):
    """This block uses idea from TGAN to generate a image frames"""
    def __init__(self):
        super(ImageGenBlockTGAN, self)

class TemporalZBlock(nn.Module):
    """This block uses idea from TGAN to generate a temporal latent vector for each frame
    generated and maps that into another space"""
    def __init__(self, params):
        super(TemporalZBlock, self).__init__()
        map1_in_channel = params["map1_in_channel"]
        map1_out_channel = params["map1_out_channel"]
        map2_out_channel = params["map2_out_channel"]
        map3_out_channel = params["map3_out_channel"]
        map4_out_channel = params["map4_out_channel"]

        self.map1_deconv = nn.ConvTranspose1d(kernel_size=3,
                                              in_channels=map1_in_channel,
                                              out_channels=map1_out_channel,
                                              stride=1)
        self.map2_deconv = nn.ConvTranspose1d(kernel_size=4,
                                              in_channels=map1_out_channel,
                                              out_channels=map2_out_channel,
                                              padding=1,
                                              stride=2)
        self.map3_deconv = nn.ConvTranspose1d(kernel_size=4,
                                              in_channels=map2_out_channel,
                                              out_channels=map3_out_channel,
                                              padding=1,
                                              stride=2)
        self.map4_deconv = nn.ConvTranspose1d(kernel_size=4,
                                              in_channels=map3_out_channel,
                                              out_channels=map4_out_channel,
                                              padding=1,
                                              stride=2)
        self.activation = nn.ReLU()

    def forward(self, x):
        # thinking of adding one linear layer here first?
        x = x.unsqueeze(dim=2)
        x = self.map1_deconv(x)
        x = self.activation(x)
        x = self.map2_deconv(x)
        x = self.activation(x)
        x = self.map3_deconv(x)
        x = self.activation(x)
        x = self.map4_deconv(x)
        x = self.activation(x)
        return x


class ResBlockBigGANup(nn.Module):
    """Residual block inspired by the ResNet Block for BigGAN and ResNet
    input: [z0, z1, z2], where z0 is latent vector for representation,
    z1 is temporal latent vector, and z2 is the linearly transformed location indicator"""
    def __init__(self, params):
        super(ResBlockBigGANup, self).__init__()
        bn1_channels = 100  # should be the siz
        # residual blocks contain multiple paths, each indexed by the letter
        self.map1_bn = nn.BatchNorm1d(bn1_channels)
        self.map2_upsample = nn.Upsample(scale_factor=4, mode='linear')
        self.map3_conv = nn.ConvTranspose2d(kernel_size=(3, 5), in_channels=100, out_channels=50)
        self.map4_bn = nn.BatchNorm2d(bn1_channels//2)
        self.map5_conv = nn.ConvTranspose2d(kernel_size=(3, 4), in_channels=50, out_channels=1)
        # LEFT PATH
        self.map1b_upsample = nn.Upsample(scale_factor=4, mode='linear')
        self.map2b_conv = nn.ConvTranspose2d(kernel_size=(3, 5), in_channels=100, out_channels=50)
        self.map3b_conv = nn.ConvTranspose2d(kernel_size=(3, 4), in_channels=50, out_channels=1)
        # RIGHT PATH (LINEAR)
        features_in = 100   # this should be the shape of the noise plus conditioning vectors
        features_out = 100
        self.map1c_linear = nn.Linear(features_in, features_out)
        self.map2c_linear = nn.Linear(features_in, features_out)
        self.activation = nn.LeakyReLU(negative_slope=0.2)

    @staticmethod
    def add(x1, x2):
        return x1 + x2

    def forward(self, x):
        x1 = self.map1c_linear(x)
        x1 = self.map1_bn(x1)
        x1 = self.activation(x1)

        x1 = x1.unsqueeze(dim=2)
        x1 = self.map2_upsample(x1)
        x1 = x1.unsqueeze(dim=3)
        x1 = self.map3_conv(x1)

        x1 = self.map4_bn(x1)
        x1 = self.activation(x1)
        x1 = self.map5_conv(x1)

        x2 = self.map2c_linear(x)
        x2 = x2.unsqueeze(dim=2)
        x2 = self.map1b_upsample(x2)
        x2 = self.activation(x2)
        x2 = x2.unsqueeze(dim=3)
        x2 = self.map2b_conv(x2)
        x2 = self.activation(x2)
        x2 = self.map3b_conv(x2)
        # print("shape is %d", x2.shape)

        x_out = self.add(x1, x2)
        x_out = self.activation(x_out)
        return x_out

class ResBlockBigGANdown(nn.Module):
    """This is the Discriminator's resnet block. Downsamples as opposed to the generator"""
    def __init__(self, params,  initial_resblock=False):
        super(ResBlockBigGANdown, self).__init__()
        avgpool_padding = params["avgpool_padding"]
        final_block_channels = params["final_block_channels"]
        kernel = params["kernel"]
        in_chann = params["in_chann"]
        kernel_b = params["kernel_b"]
        self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.initial_resblock = initial_resblock
        # note: slight change from BigGAN to minimize loss of expressiveness
        self.map1a_conv = nn.Conv2d(in_channels=in_chann, out_channels=32, kernel_size=kernel[0])
        self.map2a_conv = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=kernel[1])    # 6 X 6 output here
        self.map3a_conv = nn.Conv2d(in_channels=64, out_channels=final_block_channels, kernel_size=kernel[2])    # 4 X 4 X channels output here
        self.map1b_cov = nn.Conv2d(in_channels=in_chann, out_channels=final_block_channels, kernel_size=kernel_b)
        self.map2b_avgpool = nn.AvgPool2d(kernel_size=2, padding=avgpool_padding)   # 4 X 4 X channels output here

    def forward(self, x):
        x_res = x   # residual to be added later
        if not self.initial_resblock:
            x = self.activation(x)
        x = self.map1a_conv(x)
        x = self.activation(x)

        x = self.map2a_conv(x)
        x = self.activation(x)

        x = self.map3a_conv(x)
        # x = self.activation(x)
        # COMPUTE RESIDUAL PASSES
        # assert x_res != x
        x_res = self.map1b_cov(x_res)
        x_res = self.activation(x_res)
        x_res = self.map2b_avgpool(x_res)
        x_out = x + x_res
        x_out = self.activation(x_out)
        # print("**** Residual Block Output shape is: {} ****".format(x_out.shape))
        return x_out


class ResnetUpsample(nn.Module):
    """This is used to sample the generated temperature maps into the desired format before pass into resnet blocks

    The time dimension for generation will be converted into the batch dimension so temporal quality will not be
    evaluated by the resnet discriminator since it was not created for time-series image distribution"""
    def __init__(self, params, initial_resblock=False):
        super(ResnetUpsample, self).__init__()



#


# class ResBlockDeepBigGAN(nn.Module):
#     """Residual block inspired by the ResNet Block for Deep BigGAN Brock et. Al"""
#     def __init__(self, params):
#         super(ResBlockDeepBigGAN, self).__init__()
#
#         res_channels = 16
#         bn1_channels = 1
#         self.map1_bn = nn.BatchNorm2d(bn1_channels)
#         # TODO: first have a 4 X 4 X 16 ch then ResBlockUP
#         self.map2_conv = nn.Conv2d(in_channels=res_channels, out_channels=res_channels, kernel_size=3)
#         self.map3_bn = nn.BatchNorm2d()
#         self.map4_upsample = nn.Upsample(scale_factor=2, mode='bilinear')
#         self.map5_conv = nn.Conv2d(kernel_size=3, in_channels=1, out_channels=1)
#         self.map6_bn = nn.BatchNorm2d()
#         self.map7_conv = nn.Conv2d(kernel_size=3)
#         self.map8_bn = nn.BatchNorm2d()
#         self.map9_conv = nn.Conv2d(kernel_size=1, )
#
#     def forward(self, x):
#         torch.relu(x)
#         x = self.map1_bn(x)
#         x = self.map3_upsample(x)

#
# class SpatialTransform(nn.Module):
#     def __int__(self):
#         super()