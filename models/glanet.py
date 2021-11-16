## Code for Our full model with probability distributions

# The code borrows enoceders and decoders from HiDT
# HiD source: https://github.com/saic-mdal/HiDT

import argparse
import glob
import os
import sys

sys.path.append("..")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from typing import List, Union
from easydict import EasyDict as edict

# Helper functions for HiDT

# content encoder
from typing import List
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange, Reduce
from functools import partial

class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero', style_dim=3, norm_after_conv='ln',
                 res_off=False):
        super().__init__()
        self.res_off = res_off
        model = []
        model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type,
                              style_dim=style_dim, norm_after_conv=norm_after_conv)]
        model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type,
                              style_dim=style_dim, norm_after_conv=norm_after_conv)]
        self.model = nn.ModuleList(model)

    def forward(self, x, spade_input=None):
        residual = x
        for layer in self.model:
            x = layer(x, spade_input)
        if self.res_off:
            return x
        else:
            return x + residual


class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero', non_local=False,
                 style_dim=3, norm_after_conv='ln'):
        super(ResBlocks, self).__init__()
        self.model = []
        if isinstance(non_local, (list,)):
            for i in range(num_blocks):
                if i in non_local:
                    raise DeprecationWarning()
                else:
                    self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type,
                                            style_dim=style_dim, norm_after_conv=norm_after_conv)]
        else:
            for i in range(num_blocks):
                self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type,
                                        style_dim=style_dim, norm_after_conv=norm_after_conv)]

        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero', style_dim=3, norm_after_conv='ln'):
        super().__init__()
        self.use_bias = True
        self.norm_type = norm
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        self.compute_kernel = True if norm == 'conv_kernel' else False
        self.WCT = True if norm == 'WCT' else False

        norm_dim = output_dim

        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'WCT':
            self.norm = nn.InstanceNorm2d(norm_dim)
            self.style_dim = style_dim
            self.dim = output_dim, input_dim, kernel_size, kernel_size
            self.output_dim = output_dim
            self.stride = stride
            self.mlp_W = nn.Sequential(
                nn.Linear(self.style_dim, output_dim ** 2),
            )
            self.mlp_bias = nn.Sequential(
                nn.Linear(self.style_dim, output_dim),
            )
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        elif norm == 'conv_kernel':
            self.style_dim = style_dim
            self.norm_after_conv = norm_after_conv
            self._get_norm(self.norm_after_conv, norm_dim)
            self.dim = output_dim, input_dim, kernel_size, kernel_size
            self.stride = stride
            self.mlp_kernel = nn.Linear(self.style_dim, int(np.prod(self.dim)))
            self.mlp_bias = nn.Linear(self.style_dim, output_dim)
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if norm == 'sn':
            self.conv = SpectralNorm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

        self.style = None

    def _get_norm(self, norm, norm_dim):
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

    def forward(self, x, spade_input=None):
        if self.compute_kernel:
            conv_kernel = self.mlp_kernel(self.style)
            conv_bias = self.mlp_bias(self.style)
            x = F.conv2d(self.pad(x), conv_kernel.view(*self.dim), conv_bias.view(-1), self.stride)
        else:
            x = self.conv(self.pad(x))
        if self.WCT:
            x_mean = x.mean(-1).mean(-1)
            x = x.permute(0, 2, 3, 1)
            x = x - x_mean
            W = self.mlp_W(self.style)
            bias = self.mlp_bias(self.style)
            W = W.view(self.output_dim, self.output_dim)
            x = x @ W
            x = x + bias
            x = x.permute(0, 3, 1, 2)
        if self.norm:
            if self.norm_type == 'spade':
                x = self.norm(x, spade_input)
            else:
                x = self.norm(x)
        if self.activation:
            x = self.activation(x)

        return x


class ContentEncoderBase(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.model_preparation = nn.ModuleList()
        self.model_downsample = nn.ModuleList()
        self.model_postprocess = nn.ModuleList()

        self.output_dim = dim

    def forward(self, tensor, spade_input=None):
        model = chain(self.model_preparation, self.model_downsample, self.model_postprocess)
        return module_list_forward(model, tensor, spade_input)


class ContentEncoderBC(ContentEncoderBase):
    def __init__(self, num_downsamples, num_blocks, input_dim, dim, norm, activ, pad_type, non_local=False, **kwargs):
        super().__init__(dim)
        self.model_preparation += [Conv2dBlock(input_dim, dim, 9, 1, 4, norm=norm, activation=activ, pad_type=pad_type)]
        # downsampling blocks
        for i in range(num_downsamples):
            self.model_downsample += [
                Conv2dBlock(dim, 2 * dim, 6, 2, 2, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        # residual blocks
        self.model_postprocess += [
            ResBlocks(num_blocks, dim, norm=norm, activation=activ, pad_type=pad_type, non_local=non_local)]


class ContentEncoderUnet(ContentEncoderBC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip_dim = kwargs['skip_dim']
        if isinstance(self.skip_dim, int):
            self.skip_dim = [self.skip_dim] * kwargs['num_downsamples']

    def forward(self, tensor: torch.Tensor):
        output: List[torch.Tensor] = []
        for layer in self.model_preparation:
            tensor = layer(tensor)
        # tensor = module_list_forward(self.model_preparation, tensor, spade_input)

        for layer in self.model_downsample:
            skip_dim = 5
            if skip_dim > 0:
                out = tensor[:, :skip_dim]
            else:
                out = tensor
            output.append(out)
            tensor = layer(tensor)

        for layer in self.model_postprocess:
            tensor = layer(tensor)
        # tensor = module_list_forward(self.model_postprocess, tensor, spade_input)
        output.append(tensor)
        output_reversed: List[torch.Tensor] = [output[2], output[1], output[0]]
        return output_reversed


# style encoder
class StyleEncoderBase(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.output_dim = dim
        self.body = nn.ModuleList()
        self.head = nn.ModuleList()

    def forward(self, tensor, spade_input=None):
        if spade_input:
            for layer in self.body:
                tensor = layer(tensor, spade_input)
        else:
            for layer in self.body:
                tensor = layer(tensor)

        for layer in self.head:
            tensor = layer(tensor)

        return tensor


class StyleEncoder(StyleEncoderBase):
    def __init__(self, num_downsamples, input_dim, dim, output_dim, norm, activ, pad_type, normalized_out=False):
        super().__init__(dim)
        self.body += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        for i in range(2):
            self.body += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        for i in range(num_downsamples - 2):
            self.body += [Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]

        self.head += [nn.AdaptiveAvgPool2d(1)]
        self.head += [nn.Conv2d(dim, output_dim, 1, 1, 0)]
        if normalized_out:
            self.head += [NormalizeOutput(dim=1)]


# decoder
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim, num_blocks, norm='none', activ='relu'):
        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(input_dim, dim, norm=norm, activation=activ)]
        for i in range(num_blocks - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.model += [LinearBlock(dim, output_dim, norm='none', activation='none')]  # no output activations
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))


class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        if norm == 'sn':
            self.fc = SpectralNorm(nn.Linear(input_dim, output_dim, bias=use_bias))
        else:
            self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b).type_as(x)
        running_var = self.running_var.repeat(b).type_as(x)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


def module_list_forward(module_list: nn.ModuleList, tensor: torch.Tensor,
                        spade_input=torch.zeros(1)):
    if spade_input:
        for layer in module_list:
            tensor = layer(tensor, spade_input)
    else:
        for layer in module_list:
            tensor = layer(tensor)

    return tensor


class DecoderBase(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.body = nn.ModuleList()
        self.upsample_head = nn.ModuleList()

        self._init_modules(**kwargs)

    def _init_modules(self, **kwargs):
        raise NotImplementedError

    def forward(self, tensor, spade_input=None):
        tensor = module_list_forward(self.body, tensor, spade_input)

        for layer in self.upsample_head:
            tensor = layer(tensor)

        return tensor


class DecoderAdaINBase(DecoderBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        adain_net_config = kwargs['adain_net']
        architecture = adain_net_config.pop('architecture')
        num_adain_params = self._calc_adain_params()
        adain_net_config['output_dim'] = num_adain_params
        print('output dim:', num_adain_params)

        # self.adain_net = MLP(input_dim=3, output_dim=3, dim=64, num_blocks=3) #getattr(hidt.networks.blocks.modules, architecture)(**adain_net_config)
        self.adain_net = MLP(**adain_net_config)
        self.style_dim = adain_net_config['input_dim']
        self.pred_adain_params = 'adain' == kwargs['res_norm'] or 'adain' == kwargs['up_norm'] or 'adain' == kwargs[
            'norm_after_conv']

    def _calc_adain_params(self):
        return self.get_num_adain_params(self)

    @staticmethod
    def get_num_adain_params(model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ in ("AdaptiveInstanceNorm2d", 'AdaLIN'):
                num_adain_params += 2 * m.num_features
        return num_adain_params

    @staticmethod
    def assign_adain_params(adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ in ('AdaptiveInstanceNorm2d', 'AdaLIN'):
                assert adain_params.shape[1]
                mean = adain_params[:, :m.num_features]
                assert mean.shape[1]
                std = adain_params[:, m.num_features:2 * m.num_features]
                assert std.shape[1]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) >= 2 * m.num_features:
                    adain_params = adain_params[:, 2 * m.num_features:]

    def forward(self, content_tensor, style_tensor, spade_input=None):
        if self.pred_adain_params:
            adain_params = self.adain_net(style_tensor)
            self.assign_adain_params(adain_params, self)
        return super().forward(content_tensor, spade_input)


class DecoderAdaINConvBase(DecoderAdaINBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pred_conv_kernel = 'conv_kernel' == kwargs['res_norm'] or 'conv_kernel' == kwargs['up_norm'] or 'WCT' == \
                                kwargs['res_norm']

    @staticmethod
    def assign_style(style, model):
        # assign a style to the Conv2dBlocks
        for m in model.modules():
            if m.__class__.__name__ == "Conv2dBlock":
                m.style = style

    def forward(self, content_tensor, style_tensor, spade_input=None):
        if self.pred_conv_kernel:
            assert style_tensor.size(0) == 1, 'prediction of convilution does not work with batch size > 1'
            self.assign_style(style_tensor.view(1, -1), self)
        return super().forward(content_tensor, style_tensor, spade_input)


class DecoderUnet(DecoderAdaINConvBase):
    def _init_modules(self, **kwargs):
        self.num_upsamples = kwargs['num_upsamples']
        self.body += [ResBlocks(kwargs['num_blocks'],
                                kwargs['dim'],
                                norm=kwargs['res_norm'],
                                activation=kwargs['activ'],
                                pad_type=kwargs['pad_type'],
                                style_dim=kwargs.get('style_dim', 3))]

        self.upsample_postprocess = nn.ModuleList()
        self.skip_preprocess = nn.ModuleList()

        dim = kwargs['dim']
        skip_dim = kwargs['skip_dim']
        if isinstance(skip_dim, int):
            skip_dim = [skip_dim] * kwargs['num_upsamples']
        skip_dim = skip_dim[::-1]

        for i in range(kwargs['num_upsamples']):
            self.upsample_head += [nn.Upsample(scale_factor=2)]
            current_upsample_postprocess = [
                Conv2dBlock(dim + skip_dim[i],
                            dim // 2, 7, 1, 3,
                            norm=kwargs['up_norm'],
                            activation=kwargs['activ'],
                            pad_type=kwargs['pad_type'],
                            style_dim=kwargs.get('style_dim', 3),
                            norm_after_conv=kwargs.get('norm_after_conv', 'ln'),
                            )]
            if kwargs['num_res_conv']:
                current_upsample_postprocess += [ResBlocks(kwargs['num_res_conv'],
                                                           dim // 2,
                                                           norm=kwargs['up_norm'],
                                                           activation=kwargs['activ'],
                                                           pad_type=kwargs['pad_type'],
                                                           style_dim=kwargs.get('style_dim', 3),
                                                           norm_after_conv=kwargs.get('norm_after_conv', 'ln'),
                                                           )]

            current_skip_preprocess = [Conv2dBlock(skip_dim[i],
                                                   skip_dim[i], 7, 1, 3,
                                                   norm=kwargs['res_norm'],
                                                   activation=kwargs['activ'],
                                                   pad_type=kwargs['pad_type'],
                                                   style_dim=kwargs.get('style_dim', 3),
                                                   norm_after_conv=kwargs.get('norm_after_conv', 'ln'),
                                                   )]

            self.upsample_postprocess += [nn.Sequential(*current_upsample_postprocess)]
            self.skip_preprocess += [nn.Sequential(*current_skip_preprocess)]
            dim //= 2

        # use reflection padding in the last conv layer
        self.model_postprocess = nn.ModuleList([Conv2dBlock(dim, kwargs['output_dim'], 9, 1, 4,
                                                            norm='none',
                                                            activation='none',
                                                            pad_type=kwargs['pad_type'])])

    def forward(self, content_list, style_tensor, spade_input=None, pure_generation=False):
        if self.pred_adain_params:
            adain_params = self.adain_net(style_tensor)
            self.assign_adain_params(adain_params, self)

        if self.pred_conv_kernel:
            assert style_tensor.size(0) == 1, 'prediction of convilution does not work with batch size > 1'
            self.assign_style(style_tensor.view(1, -1), self)

        tensor = module_list_forward(self.body, content_list[0], spade_input)
        for skip_content, up_layer, up_postprocess_layer, skip_preprocess_layer in zip(content_list[1:],
                                                                                       self.upsample_head,
                                                                                       self.upsample_postprocess,
                                                                                       self.skip_preprocess):
            tensor = up_layer(tensor)
            skip_tensor = skip_preprocess_layer(skip_content)
            tensor = torch.cat([tensor, skip_tensor], 1)
            tensor = up_postprocess_layer(tensor)
        tensor = module_list_forward(self.model_postprocess, tensor, spade_input)
        return tensor


class MLP_Distribution(nn.Module):
    """
    An MLP takes in the encoded vector to predict a distribution. Might be unguided or guided distribution
    """

    def __init__(self, input_dim, latent_dim):
        super(MLP_Distribution, self).__init__()

        self.fc1 = nn.Linear(input_dim, int(0.5 * input_dim + 0.5 * latent_dim))
        self.bn1 = nn.BatchNorm1d(int(0.5 * input_dim + 0.5 * latent_dim))

        self.fc2 = nn.Linear(int(0.5 * input_dim + 0.5 * latent_dim), 2 * latent_dim)
        self.bn2 = nn.BatchNorm1d(2 * latent_dim)

        self.latent_dim = latent_dim

    def forward(self, feature, feature2=None, return_everything=False):
        # posterioir will have a feature2 as well

        if feature2 is not None:  # implying that it is guided net
            feature = torch.cat([feature, feature2], dim=1)
        ### to avoid bs=1
        feature = torch.cat([feature, feature])
        # print(feature.shape)
        encoding = self.fc1(feature)
        encoding = self.bn1(encoding)
        encoding = F.relu(encoding)

        encoding = self.fc2(encoding)
        mu_log_sigma = self.bn2(encoding)
        mu_log_sigma=mu_log_sigma[0].unsqueeze(0)
        mu = mu_log_sigma[:, :self.latent_dim]
        log_sigma = mu_log_sigma[:, self.latent_dim:]

        dist = Independent(Normal(loc=mu , scale=torch.exp(log_sigma)), 1)

        if return_everything:
            return mu, log_sigma
        else:
            return dist[0]


class Conv_Downsample(nn.Module):
    """
    Reduce spatial size and number of feature maps
    """

    def __init__(self, input_dim, output_dim):
        super(Conv_Downsample, self).__init__()

        self.conv1 = nn.Conv2d(input_dim, int(0.5 * output_dim), kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(int(0.5 * output_dim))

        self.conv2 = nn.Conv2d(int(0.5 * output_dim), output_dim, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(output_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(self.bn1(x))

        x = self.conv2(x)
        x = F.leaky_relu(self.bn2(x))

        return x

### mlp mixer
class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        return self.fn(self.norm(x)) + x

def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    return nn.Sequential(
        dense(dim, dim * expansion_factor),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(dim * expansion_factor, dim),
        nn.Dropout(dropout)
    )

def MLPMixer(image_size=256, channels=3, patch_size= 16, dim= 512, depth= 12, num_classes= 1000, expansion_factor = 4, dropout = 0.):
    assert (image_size % patch_size) == 0, 'image must be divisible by patch size'
    num_patches = (image_size // patch_size) ** 2
    chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
        nn.Linear((patch_size ** 2) * channels, dim),
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
            PreNormResidual(dim, FeedForward(dim, expansion_factor, dropout, chan_last))
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        Reduce('b n c -> b c', 'mean'),
        nn.Linear(dim, num_classes)
    )

# Our full model using HiDT modules
class glanet(nn.Module):
    def __init__(self, latent_dim=32):
        super(glanet, self).__init__()

        self.latent_dim = latent_dim
        self.style_dim = latent_dim

        self.content_encoder = ContentEncoderUnet(num_downsamples=2, num_blocks=4, input_dim=3, dim=48, norm='in',
                                                  activ='relu', pad_type='reflect', skip_dim=5)

        self.style_encoder = MLPMixer(num_classes=self.latent_dim,depth=4)
        # unguided distribution
        self.down_unguided = Conv_Downsample(192, 2)  # reduces feature map from 192x64x64 to 2x16x16=512

        self.unguided_mlp = MLP_Distribution(512 + self.latent_dim, self.latent_dim)  # 256x256


        self.guided_mlp = MLP_Distribution(self.latent_dim, self.latent_dim)

        self.decoder = DecoderUnet(num_upsamples=2, num_blocks=5, dim=192, res_norm='adain', activ='relu',
                                   pad_type='reflect', skip_dim=5, up_norm='ln', num_res_conv=0, output_dim=3,
                                   adain_net={'architecture': 'MLP', 'input_dim': self.latent_dim, 'dim': 64,
                                              'num_blocks': 3, })

    def forward(self, source, style=None, training=True):
        # print(source.shape)
        # source content
        source_content = self.content_encoder(source)

        # source style
        source_style = self.style_encoder(source)
        # print(source_style.shape)
        source_features = source_content[0]
        source_features = (self.down_unguided(source_features)).view(source.shape[0], -1)

        ### for mlp mixer
        source_features = torch.cat([source_features, source_style], dim=1)

        # print(source_features.shape)
        mu_unguided, log_var_unguided = self.unguided_mlp(source_features, return_everything=True)
        # print(mu_unguided.shape)
        # print(log_var_unguided.shape)

        unguided_distribution = Independent(Normal(loc=mu_unguided, scale=torch.exp(log_var_unguided)), 1)

        # guided synthesis: if we have a style image, we are doing guided synthesis
        if style is not None:
            # target style
            target_style = self.style_encoder(style)

            # guided distribution
            ### for mlp mixer
            mu_guided, log_var_guided = self.guided_mlp(target_style.view(source.shape[0], -1),
                                                        return_everything=True)
            guided_distribution = Independent(Normal(loc=mu_guided, scale=torch.exp(log_var_guided)), 1)

            feedback_vector = guided_distribution.rsample()
        # unguided synthesis (if there is no style image)
        else:
            feedback_vector = unguided_distribution.rsample()

        prediction = self.decoder(source_content, feedback_vector.unsqueeze(2).unsqueeze(3))

        if training:
            # print( unguided_distribution.stddev.shape)
            return nn.Tanh()(prediction), unguided_distribution.mean, unguided_distribution.stddev, guided_distribution.mean, guided_distribution.stddev, feedback_vector  # return everything during training

        else:
            return prediction
