import torch
from torch import nn, optim
import torch.nn.functional as F

from ML.layers import *

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

        self.fc1 = FC(hidden_size * 2 + scada_size, hidden_size + scada_size)
        self.fc2 = FC(hidden_size + scada_size, (hidden_size + scada_size) // 2)
        self.fc3 = FC((hidden_size + scada_size) // 2, scada_size, do=0) if one is False else FC((hidden_size + scada_size) // 2, 1, do=0)
        self.out = SigmoidRange(-1.5, 1.5)

    def _init_hidden(self, num_layers, hidden_size, batch_size, device):
        h0 = torch.zeros(num_layers * 2, batch_size, hidden_size).to(device)

        if self.rnn_type == "gru":
            return h0

        c0 = torch.zeros(num_layers * 2, batch_size, hidden_size).to(device)
        return h0, c0

    def forward(self, x):
        seq, tab = x

        out = self.layer1(seq) if self.cnn_type == "resnet" else seq
        out = self.down1(out) if self.cnn_type in ["cnn", "resnet"] else seq
        out = self.layer2(out) if self.cnn_type == "resnet" else out

        out, self.hidden = self.rnn(out, self.hidden)

        if self.attention:
            out, _ = self.atten(out)
            out = torch.cat((out, tab), 1)
        else:
            out = self.bn(out[:,-1,:])
            out = self.act(out)
            out = self.fc0(out)
            out = torch.cat((out, tab), 1)

        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.out(out)
        return out

class TabLSTMLate(nn.Module):
    def __init__(self, batch_size, input_size=30, sensor_num=21, scada_num=14, hidden_size=64, num_layers=5, dropout=0.4, device="cpu"):
        super(TabLSTMLate, self).__init__()
        self.sensor_num = sensor_num
        self.heads = nn.ModuleList(self._make_head(input_size, hidden_size, 1, dropout) for i in range(sensor_num))
        self.comb_rnn = self._make_head(input_size, hidden_size, num_layers, dropout, num_channels=(6 * sensor_num))
        self.hiddens = [self._init_hidden(num_layers, hidden_size, batch_size, device) for i in range(sensor_num)]
        self.comb_hid = self._init_hidden(num_layers, hidden_size, batch_size, device)

        feats = hidden_size * (sensor_num + 1) + scada_num
        self.fc_comb1 = nn.Linear(feats, feats)
        self.fc_comb2 = nn.Linear(feats, feats // 2)
        self.fc_comb3 = nn.Linear(feats // 2, feats // 4)
        self.fc_comb4 = nn.Linear(feats // 4, feats // 8)
        # self.fc_out = nn.Linear(feats // 8, scada_num)
        self.fc_out = nn.Linear(feats // 8, scada_num)
        self.out = SigmoidRange(-1.5, 1.5)

    def _init_hidden(self, num_layers, hidden_size, batch_size, device):
        h0 = torch.zeros(num_layers * 2, batch_size, hidden_size).to(device)
        return h0
#         c0 = torch.zeros(num_layers * 2, batch_size, hidden_size).to(device)
#         return h0, c0


    def _make_head(self, input_size, hidden_size, num_layers, dropout, num_channels=6):
        return nn.Sequential(*[
            BasicBlock1D(num_channels, num_channels),
            nn.GRU(input_size, hidden_size, num_layers, dropout=dropout, bidirectional=True, batch_first=True),
            PassHidden(nn.Sequential(*[
                TakeLast(),
                nn.BatchNorm1d(hidden_size * 2),
                nn.ReLU(),
                nn.Linear(hidden_size * 2, hidden_size)
            ]))
        ])

    def forward(self, x):
        seqs, tab = x
        outs = []

        for i in range(self.sensor_num):
            out, self.hiddens[i] = self.heads[i](seqs.split(6,1)[i])
            outs.append(out)

        out, self.comb_hid = self.comb_rnn(seqs)
        outs.append(out)

        out = torch.cat((torch.flatten(torch.stack(outs, 2), 1), tab), 1)
        out = self.fc_comb1(out)
        out = self.fc_comb2(out)
        out = self.fc_comb3(out)
        out = self.fc_comb4(out)
        out = self.fc_out(out)
        out = self.out(out)
        return out

class TabLSTM(nn.Module):
    def __init__(self, batch_size, input_size=30, sensor_num=21, scada_size=14, hidden_size=64, num_layers=5, dropout=0.3, device="cpu", one=False):
        super(TabLSTM, self).__init__()
        # self.cnn = nn.Sequential(*[BasicBlock1D(6 * sensor_num, 6 * sensor_num) for i in range(1)])
        # self.conv3 = Conv1D(6 * sensor_num, 6 * sensor_num)
        # self.conv5 = Conv1D(6 * sensor_num, 6 * sensor_num, ks=5, pad=2)
        # self.conv1 = Conv1D(6 * sensor_num * 3, 6 * sensor_num * 3 // 2, ks=1, pad=0)
        #
        self.layer0 = nn.Sequential(*[BasicBlock1D(6 * sensor_num, 6 * sensor_num) for i in range(2)])
        self.down1 = BasicBlock1D(6 * sensor_num, 12 * sensor_num, downsample=True)
        self.layer1 = nn.Sequential(*[BasicBlock1D(12 * sensor_num, 12 * sensor_num) for i in range(2)])

        self.lstm = nn.GRU(input_size // 2 + 1, hidden_size, num_layers, dropout=dropout, bidirectional=True, batch_first=True)
        self.hidden = self._init_hidden(num_layers, hidden_size, batch_size, device)
        self.bn = nn.BatchNorm1d(hidden_size * 2)
        self.act = nn.ReLU()
        self.atten = Attention(hidden_size * 2)

        # self.fc1 = FC(hidden_size * 2, hidden_size)
        # self.fc3 = FC(hidden_size, hidden_size // 2)
        # self.fc4 = FC(hidden_size // 2, scada_size, do=0)
        self.fc1 = FC(hidden_size * 2 + scada_size, hidden_size + scada_size)
        # self.fc2 = FC(hidden_size + scada_size, hidden_size + scada_size)
        self.fc3 = FC(hidden_size + scada_size, (hidden_size + scada_size) // 2)
        self.fc4 = FC((hidden_size + scada_size) // 2, scada_size, do=0) if one is False else FC((hidden_size + scada_size) // 2, 1, do=0)
        self.out = SigmoidRange(-1.5, 1.5)

    def _init_hidden(self, num_layers, hidden_size, batch_size, device):
        h0 = torch.zeros(num_layers * 2, batch_size, hidden_size).to(device)
        c0 = torch.zeros(num_layers * 2, batch_size, hidden_size).to(device)
        # return h0, c0
        return h0

    def forward(self, x):
        seq, tab = x
        # seq = self.cnn(seq)
        # seq3 = self.conv3(seq)
        # seq5 = self.conv5(seq)
        # seq = torch.cat((seq, seq3, seq5), 1)
        # seq = self.conv1(seq)

        out = self.layer0(seq)
        out = self.down1(out)
        out = self.layer1(out)

        out, self.hidden = self.lstm(out, self.hidden)
        out, _ = self.atten(out)
        # out = self.bn(out)
        # out = self.act(out)
#         out = self.fc1(out)

        # out = torch.cat((out[:,-1,:], tab), 1)
        out = torch.cat((out, tab), 1)
        out = self.fc1(out)
#         out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        out = self.out(out)
        return out

class CNN(nn.Module):
    def __init__(self, input_size=30, sensor_num=21, scada_size=14, hidden_size=64, num_layers=5, dropout=0.3, device="cpu"):
        super(CNN, self).__init__()
        self.layer0 = nn.Sequential(*[BasicBlock1D(6 * sensor_num, 6 * sensor_num) for i in range(2)])
        self.down1 = BasicBlock1D(6 * sensor_num, 12 * sensor_num, downsample=True)
        self.layer1 = nn.Sequential(*[BasicBlock1D(12 * sensor_num, 12 * sensor_num) for i in range(2)])
        self.down2 = BasicBlock1D(12 * sensor_num, 24 * sensor_num, downsample=True)
        self.layer2 = nn.Sequential(*[BasicBlock1D(24 * sensor_num, 24 * sensor_num) for i in range(2)])
        self.down3 = BasicBlock1D(24 * sensor_num, 48 * sensor_num, downsample=True)
        self.layer3 = nn.Sequential(*[BasicBlock1D(48 * sensor_num, 48 * sensor_num) for i in range(2)])

        self.fc1 = FC(48 * sensor_num * 6 + scada_size, 12 * sensor_num * 6)
        self.fc2 = FC(12 * sensor_num * 6, 3 * sensor_num * 6)
        self.fc3 = FC(3 * sensor_num * 6, sensor_num * 6, do=0)
        # self.fc4 = FC(sensor_num * 6, scada_size, do=0)
        self.fc4 = FC(sensor_num * 6, scada_size, do=0)
        self.out = SigmoidRange(-1.5, 1.5)

    def _init_hidden(self, num_layers, hidden_size, batch_size, device):
        h0 = torch.zeros(num_layers * 2, batch_size, hidden_size).to(device)
#         c0 = torch.zeros(num_layers * 2, batch_size, hidden_size).to(device)
#         return h0, c0
        return h0

    def forward(self, x):
        seq, tab = x

        out = self.layer0(seq)
        out = self.down1(out)
        out = self.layer1(out)
        out = self.down2(out)
        out = self.layer2(out)
        out = self.down3(out)
        out = self.layer3(out)
        out = torch.flatten(out, start_dim=1)

        out = torch.cat((out, tab), 1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        out = self.out(out)
        return out