from turtle import forward
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

from config import Config
from .base import BaseModel
import ipdb


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        return self.layers(x)


class Model(BaseModel):
    def __init__(self, config: Config):
        super().__init__(config)

    def build_model(self):
        self.encoder = Encoder(
            self.config.input_dim, self.config.hidden_dim
        )
        self.out = nn.Linear(self.config.hidden_dim, 1)

    def forward(self, inputs):
        hidden = self.encoder(inputs)
        return self.out(hidden)
    

    def train_step(self, inputs, outputs, epoch):
        # score = self.calc_student_rec_loss(batch, flag='train')
        logits = self(inputs)
            
        loss = F.mse_loss(logits, outputs)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.running_loss += loss.item()
        self.iterations += 1

    def val_step(self, inputs, outputs, epoch):
        logits = self(inputs)
        loss = F.mse_loss(logits, outputs)
        return loss

    def decision_function(self, inputs, pseudo_labels, epoch):
        logits = self(inputs).to('cpu')
        pseudo_labels = torch.cat([pseudo_labels, logits], dim=-1)

        std = pseudo_labels.std(dim=-1, keepdim=True)

        if self.config.experiment_type == 'uadb':
            return logits + std
        elif self.config.experiment_type == 'base_mean':
            return logits
        elif self.config.experiment_type == 'base_std':
            return std
        elif self.config.experiment_type == 'base_mean_cascade':
            return logits, logits
        elif self.config.experiment_type == 'base_std_cascade':
            return logits, std
