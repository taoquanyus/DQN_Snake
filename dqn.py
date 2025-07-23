import torch
from torch import nn, optim

from static_parameters import LR, GAMMA


class DQN(nn.Module):
    def __init__(self, model, device):
        super().__init__()
        self.lr = LR
        self.gamma = GAMMA
        self.device = device
        self.model = model.to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        # 把state都用tensor来表示
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        next_state = torch.tensor(next_state, dtype=torch.float, device=self.device)
        action = torch.tensor(action, dtype=torch.long, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float, device=self.device)

        if len(state.shape) == 1: # 短期记忆
            state = torch.unsqueeze(state, 0)
            if not done:
                next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        predict = self.model(state)

        target = predict.clone()

        for i in range(len(done)):
            if done[i]:
                Q_new = reward[i]
            else:
                Q_new = reward[i] + self.gamma * torch.max(self.model(torch.unsqueeze(next_state[i], 0)))

            target[i][action[i]] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, predict)
        loss.backward()     #反向传播更新参数
        self.optimizer.step()