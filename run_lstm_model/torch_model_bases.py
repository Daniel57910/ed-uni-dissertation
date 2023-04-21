import torch 
from torch import nn
N_FEATURES = 18
class LSTMOrdinal(nn.Module):
    def __init__(self,  hidden_size=32, dropout=0.2) -> None:
        super(LSTMOrdinal, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=N_FEATURES,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        self.output = nn.Linear(
            hidden_size,
            1
        )

    def forward(self, x):


        x, _ = self.lstm(x)
        x = x[:, -1]
        return self.output(x)