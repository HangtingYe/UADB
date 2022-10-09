import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import average_precision_score, roc_auc_score

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


import torch
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.dataloader import DataLoader

from data_generator import DataGenerator


logger = logging.getLogger(__name__)


# def load_data(data_path, label_col='class'):
#     raw = pd.read_csv(data_path)

#     fea_cols = list(raw.columns)
#     fea_cols.remove(label_col)

#     inputs = raw[fea_cols].values
#     outputs = raw[[label_col]].values

#     return inputs, outputs


def load_data(config):
    data_generator = DataGenerator(dataset=config.data_path)

    assert (config.realistic_synthetic_mode != 'none') + (config.noise_type != 'none') < 2

    if config.realistic_synthetic_mode != 'none':
        data = data_generator.generator(
            la=1.0,
            realistic_synthetic_mode=config.realistic_synthetic_mode,
            noise_ratio=config.noise_ratio
        )
    elif config.noise_type != 'none':
        data = data_generator.generator(
            la=1.0,
            noise_type=config.noise_type,
            noise_ratio=config.noise_ratio,
            duplicate_times=config.duplicate_times
        )
    else:
        data = data_generator.generator(la=1.0)

    return data['X_train'], data['y_train']


class TabularData(object):
    def __init__(self, config) -> None:
        self.config = config
        self.inputs, self.outputs = load_data(self.config)
        self.input_dim = self.inputs.shape[1]

        self.pseudo_labels = self.get_init_labels()

        metrics = {
            'aucroc': roc_auc_score(self.outputs, self.pseudo_labels),
            'ap': average_precision_score(self.outputs, self.pseudo_labels)
        }

        logger.info(
            f'Init Performance on Dataset {self.config.data_path}, AUC ROC {metrics["aucroc"]}, AP {metrics["ap"]}'
        )

        self.build_entire_dataloader()
        self.build_train_dataloaders()

    def get_init_labels(self):
        pseudo_models = {'pca':PCA(), 'iforest':IForest(), 'hbos':HBOS(), 'ocsvm':OCSVM(), 'lof':LOF(), 'cblof':CBLOF(), 'cof':COF(), 'knn':KNN(), 'sod':SOD(), 'ecod':ECOD(), 'deep_svdd':DeepSVDD(), 'loda':LODA(), 'copod':COPOD(), 'gmm':GMM()}
        # model = IForest()
        model = pseudo_models[self.config.pseudo_model]
        model.fit(self.inputs)
        score = model.decision_function(self.inputs)
        score = MinMaxScaler().fit_transform(score.reshape(-1, 1))
        return score

    def update_pseudo_label(self, scores):
        scores = np.expand_dims(scores, axis=-1)
        self.pseudo_labels = np.concatenate(
            [self.pseudo_labels, scores], axis=-1
        )

    def build_entire_dataloader(self):
        inputs = torch.from_numpy(self.inputs).float()
        outputs = torch.from_numpy(self.outputs).float()
        pseudo_labels = torch.from_numpy(self.pseudo_labels).float()

        dataset = TensorDataset(inputs, outputs, pseudo_labels)
        self.entire_loader = self._build_dataloader(dataset, shuffle=False)

    def build_train_dataloaders(self):
        kf = KFold(n_splits=3)

        self.train_loaders = []
        self.val_loaders = []

        for train_idx, val_idx in kf.split(self.inputs):
            train_inputs = torch.from_numpy(self.inputs[train_idx]).float()
            train_outputs = torch.from_numpy(self.outputs[train_idx]).float()
            train_pseudo_labels = torch.from_numpy(self.pseudo_labels[train_idx]).float()

            self.train_loaders.append(
                self._build_dataloader(TensorDataset(train_inputs, train_outputs, train_pseudo_labels), shuffle=True)
            )

            val_inputs = torch.from_numpy(self.inputs[val_idx]).float()
            val_outputs = torch.from_numpy(self.outputs[val_idx]).float()
            val_pseudo_labels = torch.from_numpy(self.pseudo_labels[val_idx]).float()

            self.val_loaders.append(
                self._build_dataloader(TensorDataset(val_inputs, val_outputs, val_pseudo_labels), shuffle=False)
            )

    def _build_dataloader(self, dataset, shuffle=True):
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle
        )
        return dataloader
