import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter
from torch.distributions import Bernoulli
import math



class ModuleWrapper(nn.Module):
    """Wrapper for nn.Module with support for arbitrary flags and a universal forward pass"""

    def __init__(self):
        super(ModuleWrapper, self).__init__()

    def set_flag(self, flag_name, value):
        setattr(self, flag_name, value)
        for m in self.children():
            if hasattr(m, 'set_flag'):
                m.set_flag(flag_name, value)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x


class Abs(ModuleWrapper):

    def __init__(self):
        super(Abs, self).__init__()

    def forward(self, x):
        return x.abs_()


class FlattenLayer(ModuleWrapper):

    def __init__(self, num_features):
        super(FlattenLayer, self).__init__()
        self.num_features = num_features

    def forward(self, x):
        return x.view(-1, self.num_features)
    
class LinearVariance(ModuleWrapper):
    def __init__(self, in_features, out_features, bias=True, rounding = 0):
        super(LinearVariance, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rounding = rounding
        self.sigma = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.sigma.size(1))
        self.sigma.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x):
        if self.bias:
            lrt_mean = self.bias
        else:
            lrt_mean = 0.0
        lrt_std = torch.sqrt_(1e-16 + F.linear(x * x, self.sigma * self.sigma))
        if self.training:
            eps = Variable(lrt_std.data.new(lrt_std.size()).normal_())
        else:
            eps = lrt_std.data.new(lrt_std.size()).normal_()
        if self.rounding:
            eps = torch.round(eps * self.rounding)/self.rounding
        return lrt_mean + eps * lrt_std
    
    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', bias=' + str(self.bias is not None)  \
               + ', rounding='+str(self.rounding) + ')'

class LinearVarianceUnif(ModuleWrapper):

    def __init__(self, in_features, out_features, bias=True, rounding =0):
        super(LinearVarianceUnif, self).__init__()
        self.in_features = in_features
        self.rounding = rounding
        self.out_features = out_features
        self.W = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x):
        if self.training:
            eps = Variable(self.W.data.new(self.W.size()).uniform_() - 0.5)
        else:
            eps = self.W.data.new(self.W.size()).uniform_() - 0.5
        if self.rounding:
            eps = torch.round(eps * self.rounding)/self.rounding
        output = F.linear(x, self.W*eps)
        if self.bias is not None:
            output = output + self.bias
        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', bias=' + str(self.bias is not None)  \
               + ', rounding='+str(self.rounding) + ')'

class ConvVariance(ModuleWrapper):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, bias=True, rounding = 0):
        super(ConvVariance, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = 1
        self.rounding=rounding
        self.sigma = Parameter(torch.Tensor(
            out_channels, in_channels, *self.kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(1, out_channels, 1, 1))
        else:
            self.register_parameter('bias', None)
        self.op_bias = lambda input, kernel: F.conv2d(input, kernel, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self.op_nobias = lambda input, kernel: F.conv2d(input, kernel, None, self.stride, self.padding, self.dilation, self.groups)
        self.reset_parameters()
        self.zero_mean = False
        self.permute_sigma = False

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.sigma.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        lrt_mean = 0.0
        if self.bias is not None:
            lrt_mean = self.bias

        sigma2 = self.sigma * self.sigma
        if self.permute_sigma:
            sigma2 = sigma2.view(-1)[torch.randperm(self.weight.shape).cuda()].view(self.weight.shape)

        lrt_std = Variable(torch.sqrt(1e-16 + self.op_nobias(x * x, sigma2)))
        if self.training:
            eps = Variable(lrt_std.data.new(lrt_std.size()).normal_())
        else:
            eps = lrt_std.data.new(lrt_std.size()).normal_()
        if self.rounding:
            eps=torch.round(eps*self.rounding)/self.rounding
        return lrt_mean + lrt_std * eps

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        s += ', padding={padding}'
        s += ', dilation={dilation}'
        s += ', rounding={rounding}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

class ConvVarianceUnif(ModuleWrapper):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, bias=True, rounding = 0):
        super(ConvVarianceUnif, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = 1
        self.rounding = rounding
        self.W = Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(1, out_channels, 1, 1))
        else:
            self.register_parameter('bias', None)
        self.op_bias = lambda input, kernel: F.conv2d(input, kernel, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self.op_nobias = lambda input, kernel: F.conv2d(input, kernel, None, self.stride, self.padding, self.dilation, self.groups)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.W.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        if self.training:
            eps = Variable(torch.rand(self.W.size()) - 0.5)
        else:
            eps = torch.rand(self.W.size()) - 0.5
        if self.rounding:
            eps=torch.round(eps*self.rounding)/self.rounding
        eps=eps.cuda()
        output = self.op_nobias(x, self.W*eps)
        if self.bias is not None:
            output = output + self.bias
        return output

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        s += ', padding={padding}'
        s += ', dilation={dilation}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)
