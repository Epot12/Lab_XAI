import torch.nn as nn

class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # out: (batch_size, seq_len, hidden_dim)
        out, _ = self.lstm(x)
        # Prendiamo solo l'output dell'ultimo step della sequenza (il giorno corrente)
        out = out[:, -1, :]
        return self.linear(out)