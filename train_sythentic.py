import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pyod.models.iforest import IForest
from pyod.models.hbos import HBOS
from pyod.models.pca import PCA
from pyod.models.ocsvm import OCSVM
from pyod.models.lof import LOF
from pyod.models.cblof import CBLOF
from pyod.models.cof import COF
from pyod.models.knn import KNN
from pyod.models.sod import SOD
from pyod.models.ecod import ECOD
from pyod.models.deep_svdd import DeepSVDD
from pyod.models.loda import LODA
from pyod.models.copod import COPOD
from pyod.models.gmm import GMM
from sklearn.preprocessing import MinMaxScaler
import torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from numpy import percentile
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.dataloader import DataLoader
from copulas.multivariate import VineCopula
from copulas.univariate import GaussianKDE

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


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.build_model()
        self.optimizer = optim.Adam(
            self.parameters(), lr=1e-3,
        )

    def build_model(self):
        self.encoder = Encoder(
            2, 128
        )
        self.out = nn.Linear(128, 1)

    def forward(self, inputs):
        hidden = self.encoder(inputs)
        return self.out(hidden)
    

    def train_step(self, inputs, outputs):
        # score = self.calc_student_rec_loss(batch, flag='train')
        logits = self(inputs)
            
        loss = F.mse_loss(logits, outputs)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def val_step(self, inputs, outputs):
        logits = self(inputs)
        loss = F.mse_loss(logits, outputs)
        return loss

    def decision_function(self, inputs, pseudo_labels):
        logits = self(inputs).to('cpu')
        pseudo_labels = torch.cat([pseudo_labels, logits], dim=-1)

        std = pseudo_labels.std(dim=-1, keepdim=True)

        return logits + std


# Define the number of inliers and outliers
n_samples = 320
outliers_fraction = 0.1

xx, yy = np.meshgrid(np.linspace(-7, 7, 100), np.linspace(-7, 7, 100))

n_inliers = int((1. - outliers_fraction) * n_samples)
n_outliers = int(outliers_fraction * n_samples)
ground_truth = np.zeros(n_samples, dtype=int)
ground_truth[-n_outliers:] = 1
random_state = np.random.RandomState(42)

# Data generation
X1 = 0.6 * np.random.randn(n_inliers // 4, 2)
X1[:, 0] = X1[:, 0]-3
X1[:, 1] = X1[:, 1]+3
X2 = 0.6 * np.random.randn(n_inliers // 4, 2)
X3 = 0.6 * np.random.randn(n_inliers // 4, 2)
X3[:, 0] = X3[:, 0]+2.5
X3[:, 1] = X3[:, 1]+3
X4 = 0.6 * np.random.randn(n_inliers // 4, 2)
X4[:, 0] = X4[:, 0]+3.5
X4[:, 1] = X4[:, 1]-3.5


X = np.r_[X1, X2, X3, X4]


# Add outliers
# X (200, 2)

# X = np.r_[X, np.random.uniform(low=-6, high=6, size=(n_outliers, 2))]

# X5 = 0.5 * np.random.randn(n_outliers, 2)
# X5[:, 0] = X5[:, 0]-4.5
# X5[:, 1] = X5[:, 1]-4.5
# X = np.r_[X, X5]

X5 = 0.8 * np.random.randn(n_outliers // 4, 2)
X5[:, 0] = X5[:, 0]-3.5
X5[:, 1] = X5[:, 1]+2.5
X6 = 0.8 * np.random.randn(n_outliers // 4, 2)
X7 = 0.8 * np.random.randn(n_outliers // 4, 2)
X7[:, 0] = X7[:, 0]+3.5
X7[:, 1] = X7[:, 1]+3
X8 = 0.8 * np.random.randn(n_outliers // 4, 2)
X8[:, 0] = X8[:, 0]+3.5
X8[:, 1] = X8[:, 1]-3.5

X = np.r_[X, X5, X6, X7, X8]



pseudo_labels = []
boundary_scores = []

# fit the data and tag outliers
teacher = IForest()
teacher.fit(X)

score = teacher.decision_function(X).reshape(-1,1)
b_score = teacher.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(-1,1)
scaler = MinMaxScaler().fit(np.concatenate([score, b_score], axis=0))
score = scaler.transform(score).reshape(-1,1)
b_score = scaler.transform(b_score).reshape(-1,1)
pseudo_labels.append(score)
boundary_scores.append(b_score)

student = Model()
student = student.cuda()
student.train()
for rep in range(40):
    dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(pseudo_labels[-1]).float())
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    for epoch in range(20):
        running_loss = 0.0
        for iter, (train_x, train_y) in enumerate(dataloader):
            train_x, train_y = train_x.cuda(), train_y.cuda()
            loss = student.train_step(train_x, train_y)
            running_loss += loss.item()
        print(f'Epoch {epoch} loss: {running_loss / len(dataloader)}')

    score = student.decision_function(torch.from_numpy(X).float().cuda(), torch.from_numpy(np.concatenate(pseudo_labels, axis=-1)).float())
    score = score.data.numpy().reshape(-1,1)

    b_score = student.decision_function(torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float().cuda(), torch.from_numpy(np.concatenate(boundary_scores, axis=-1)).float())
    b_score = b_score.data.numpy().reshape(-1,1)

    scaler = MinMaxScaler().fit(np.concatenate([score, b_score], axis=0))
    score = scaler.transform(score).reshape(-1,1)
    b_score = scaler.transform(b_score).reshape(-1,1)
    pseudo_labels.append(score)
    boundary_scores.append(b_score)

type_ = 'iforest_local'    
np.save(f'./results_sythentic/{type_}/X', X)
np.save(f'./results_sythentic/{type_}/xx', xx)
np.save(f'./results_sythentic/{type_}/yy', yy)
np.save(f'./results_sythentic/{type_}/pseudo_labels', pseudo_labels)
np.save(f'./results_sythentic/{type_}/boundary_scores', boundary_scores)
np.save(f'./results_sythentic/{type_}/n_samples', n_samples)
np.save(f'./results_sythentic/{type_}/outliers_fraction', outliers_fraction)

