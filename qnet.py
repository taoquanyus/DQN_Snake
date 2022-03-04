import torch
from torch import nn

from static_parameters import STATE_DIM, ACTION_DIM


class Qnet(nn.Module):
    def __init__(self):
        self.state_dim = STATE_DIM
        self.action_dim = ACTION_DIM
        self.train_score = []
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Linear(self.state_dim, 64, bias=True),
            nn.ReLU()
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(64, 16, bias=True),
            nn.ReLU()
        )
        self.out = nn.Sequential(
            nn.Linear(16, self.action_dim, bias=True)
        )

    def init_weights(self, m):
        if type(m) == nn.Conv2d:
            m.weight.data.normal_(0.01, 0.02)
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        x = self.hidden(x)
        x = self.hidden1(x)
        x = self.out(x)
        return x
