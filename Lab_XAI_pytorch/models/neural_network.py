import torch.nn as nn

class NeurNet(nn.Module):
    def __init__(self, input_dim):
        super(NeurNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 800),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(800, 400),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(400, 200),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, 1),
            nn.ReLU()
        )

        # Apply Keras style initialization at build time
        self.apply(self._init_weights_keras)

    def _init_weights_keras(self, m):
        if isinstance(m, nn.Linear):
            # Keras default: Glorot / Xavier Uniform for the weights
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                # Keras default: Zeros for biases
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)