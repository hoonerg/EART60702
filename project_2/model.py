import torch
import torch.nn as nn

class TimeSeriesNN(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(TimeSeriesNN, self).__init__()
        self.layer1 = nn.Linear(570, 128)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(128, 128)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.layer3 = nn.Linear(128, 64)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.layer4 = nn.Linear(64, 32)
        self.dropout4 = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.dropout1(x)
        x = torch.relu(self.layer2(x))
        x = self.dropout2(x)
        x = torch.relu(self.layer3(x))
        x = self.dropout3(x)
        x = torch.relu(self.layer4(x))
        x = self.dropout4(x)
        x = self.output_layer(x)
        return x