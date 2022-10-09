from typing import List
import torch
import torch.nn as nn
import torch.optim as optim

from config import Config


class BaseModel(nn.Module):
    """Define the basic APIs"""

    def __init__(self, config: Config):
        super().__init__()

        self.config = config
        self.build_model()
        # self.optimizer = self.get_optimizer()
        self.get_optimizer()

    def build_model(self):
        raise NotImplementedError

    def train_step(self, batch, epoch):
        score = self(batch)
        loss = torch.mean(score)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.running_loss += loss.item()
        self.iterations += 1

    def train_epoch_start(self):
        self.running_loss = 0.0
        self.iterations = 0

    def train_epoch_end(self):
        # return training message
        msg = f'Average Error: {self.running_loss / self.iterations:.4f}'
        return msg

    def val_step(self, batch, epoch):
        score = self(batch)
        return score

    def get_optimizer(self):
        self.optimizer = optim.Adam(
            self.parameters(), lr=self.config.learning_rate,
        )
