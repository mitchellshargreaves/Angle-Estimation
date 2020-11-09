import torch
from torch import nn, optim
import torch.nn.functional as F

from ML.layers import *

# SensorRNN constructor
# batch_size: training batch size
# input_size: length of the timeseries input
# sensor_num: the number of PMU sensors
# scada_size: the number of SCADA sensors
# rnn_type: "gru" or "lstm"
# cnn_type: "cnn" (simple CNN-RNN), "resnet" (adds 1D basic blocks) or None (No CNN layers)
# attention: boolean, if true, activate RNN with attention else BN, ReLU and FC
# hidden_size: RNN hidden layer dimensiom
# num_layers: number of RNN layers
# dropout: RNN dropout
# device: Device to put hidden layers on
# one: boolean, if true, return one angle estimation, else return scada_size estimations

# Uses a CNN-RNN with attention to process PMU data (sensors are concatenated on the channels axis)
# RNN features are concatenated with SCADA magnitude measurements before fully connected layer processing

class SensorRNN(nn.Module):
    def __init__(self, batch_size, input_size, sensor_num, scada_size, rnn_type="gru", cnn_type="resnet", attention=False, hidden_size=64, num_layers=5, dropout=0.3, device="cpu", one=False):
        super(SensorRNN, self).__init__()
        # Store model type hyperparameters
        self.rnn_type = rnn_type
        self.cnn_type = cnn_type
        self.attention = attention

        # Make CNN layers if required
        if cnn_type == "resnet":
            self.layer1 = nn.Sequential(*[BasicBlock1D(6 * sensor_num, 6 * sensor_num) for i in range(2)])
            self.layer2 = nn.Sequential(*[BasicBlock1D(12 * sensor_num, 12 * sensor_num) for i in range(2)])

        if cnn_type is not None:
            self.down1 = BasicBlock1D(6 * sensor_num, 12 * sensor_num, downsample=True)

        # Make RNN layers
        rnn_layer = nn.GRU if rnn_type == "gru" else nn.LSTM if rnn_type == "lstm" else None
        input_size = input_size if cnn_type is None else input_size // 2 + 1

        self.rnn = rnn_layer(input_size, hidden_size, num_layers, dropout=dropout, bidirectional=True, batch_first=True)
        self.hidden = self._init_hidden(num_layers, hidden_size, batch_size, device)

        # Make RNN activation layers
        if attention:
            self.atten = Attention(hidden_size * 2)
        else:
            self.bn = nn.BatchNorm1d(hidden_size * 2)
            self.act = nn.ReLU()
            self.fc0 = FC(hidden_size * 2, hidden_size * 2)

        # Make FC layers
        self.fc1 = FC(hidden_size * 2 + scada_size, hidden_size + scada_size)
        self.fc2 = FC(hidden_size + scada_size, (hidden_size + scada_size) // 2)
        self.fc3 = FC((hidden_size + scada_size) // 2, scada_size, do=0) if one is False else FC((hidden_size + scada_size) // 2, 1, do=0)
        self.out = SigmoidRange(-1.5, 1.5)

    # Initialise RNN hidden layers
    def _init_hidden(self, num_layers, hidden_size, batch_size, device):
        h0 = torch.zeros(num_layers * 2, batch_size, hidden_size).to(device)

        if self.rnn_type == "gru":
            return h0

        c0 = torch.zeros(num_layers * 2, batch_size, hidden_size).to(device)
        return h0, c0

    # Accepts a tuple of sequential (PMU) data and tabular (SCADA magnitudes)
    def forward(self, x):
        # Seperate out sequential from tabular
        seq, tab = x

        # Pass through CNN layers
        out = self.layer1(seq) if self.cnn_type == "resnet" else seq
        out = self.down1(out) if self.cnn_type in ["cnn", "resnet"] else seq
        out = self.layer2(out) if self.cnn_type == "resnet" else out

        # Pass through RNN layers
        out, self.hidden = self.rnn(out, self.hidden)

        # Activate with attention
        if self.attention:
            out, _ = self.atten(out)
            out = torch.cat((out, tab), 1)

        # Reprocess with BN, ReLU, FC
        else:
            out = self.bn(out[:,-1,:])
            out = self.act(out)
            out = self.fc0(out)
            out = torch.cat((out, tab), 1)

        # FC output
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.out(out)
        return out