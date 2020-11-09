import numpy as np

import torch
from torch import nn, optim
import torch.nn.functional as F

class FC(nn.Module):
    def __init__(self, in_feat, out_feat, bn=True, do=.1):
        super(FC, self).__init__()
        self.fc = nn.Linear(in_feat, out_feat)
#         self.activ = nn.LeakyReLU(0.1)
        self.activ = nn.ReLU()

        if bn:
            self.bn = nn.BatchNorm1d(out_feat)
        else:
            self.bn = None

        if do > 0:
            self.do = nn.Dropout(p=do)
        else:
            self.do = None

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x) if self.bn else x
        x = self.activ(x)
        x = self.do(x) if self.do else x

        return x

class SigmoidRange(nn.Module):
    def __init__(self, high, low):
        super(SigmoidRange, self).__init__()
        self.high = high
        self.low = low

    def forward(self, x):
        return torch.sigmoid(x) * (self.high - self.low) + self.low

# Convolution, batch norm, ReLu with typical parameters defaulted
class Conv1D(nn.Module):
    def __init__(self, in_planes, out_planes, ks=3, stride=1, pad=1, bn=True, activ=nn.ReLU(inplace=True)):
        super().__init__()
        self.conv = nn.Conv1d(
            in_planes,
            out_planes,
            kernel_size=ks,
            stride=stride,
            padding=pad,
            bias=False
        )

        # Optionally include activations
        self.bn = nn.BatchNorm1d(out_planes) if bn else None
        self.activ = activ if activ else None

    def forward(self, x):
        x = self.conv(x)

        # Optionally include activations
        if self.bn:
            x = self.bn(x)

        if self.activ:
            x = self.activ(x)
        return x

# ResNet Basic Block
class BasicBlock1D(nn.Module):
    def __init__(self, in_planes, out_planes, downsample=False):
        super().__init__()

        # Downsampling uses a stride 2 conv
        self.downsample = downsample
        if self.downsample:
            self.conv1 = Conv1D(in_planes, out_planes, stride=2, pad=2)
            self.downsample = Conv1D(in_planes, out_planes, ks=1, stride=2, activ=None)
        else:
            self.conv1 = Conv1D(in_planes, out_planes)

        self.conv2 = Conv1D(out_planes, out_planes, activ=None)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Save original
        identity = x

        # Double conv
        out = self.conv1(x)
        out = self.conv2(out)

        # Add skip and activate
        if self.downsample:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out

class TakeLast(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x[:,-1,:]

class PassHidden(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def forward(self, inputs):
        x, y = inputs
        return self.layers(x), y

# https://www.kaggle.com/dannykliu/lstm-with-attention-clr-in-pytorch
class Attention(nn.Module):
    def __init__(self, hidden_size, batch_first=False):
        super(Attention, self).__init__()

        self.hidden_size = hidden_size
        self.att_weights = nn.Parameter(torch.Tensor(1, hidden_size), requires_grad=True)

        stdv = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.att_weights:
            nn.init.uniform_(weight, -stdv, stdv)

    def get_mask(self):
        pass

    def forward(self, inputs):
        # apply attention layer
        weights = torch.bmm(inputs,
                            self.att_weights  # (1, hidden_size)
                            .permute(1, 0)  # (hidden_size, 1)
                            .unsqueeze(0)  # (1, hidden_size, 1)
                            .repeat(inputs.size()[0], 1, 1) # (batch_size, hidden_size, 1)
                            )

        attentions = torch.softmax(F.relu(weights.squeeze()), dim=-1)

        # create mask based on the sentence lengths
        mask = torch.ones(attentions.size(), requires_grad=True).cuda()

        # apply mask and renormalize attention scores (weights)
        masked = attentions * mask
        _sums = masked.sum(-1).unsqueeze(-1)  # sums per row

        attentions = masked.div(_sums)

        # apply attention weights
        weighted = torch.mul(inputs, attentions.unsqueeze(-1).expand_as(inputs))

        # get the final fixed vector representations of the sentences
        representations = weighted.sum(1).squeeze()

        return representations, attentions
