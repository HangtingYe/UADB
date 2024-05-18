from statistics import mean
import copy
import random
import importlib

import numpy as np
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import logging

from data import TabularData
from config import Config
from models.base import BaseModel
from itertools import cycle

from sklearn.metrics import average_precision_score, roc_auc_score
import ipdb

logger = logging.getLogger(__name__)

read_file = 'Results_50_iter'

class Pipeline(object):
    def __init__(self, config: Config) -> None:
        logger.info('Pipeline Initialization')
        self.set_config(config)
        self.set_random_seed()
        self.get_data()

    def set_config(self, config: Config):
        self.config = config
        self.device = torch.device(self.config.device)

    def set_random_seed(self, seed=None):
        if seed is None:
            seed = self.config.seed
        else:
            self.config.seed = seed

        logger.info(f'Set random seed {seed}')
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def get_data(self):
        logger.info('Load data')
        self.tab_data = TabularData(self.config)

    def get_model(self):
        model_class = importlib.import_module(
            f'models.{self.config.model}'
        )
        self.config.input_dim = self.tab_data.input_dim

        if self.config.hidden_dim == 'auto':
            self.config.hidden_dim = self.infer_hidden_dim()

        self.model = model_class.Model(
            self.config,
        ).to(self.device)

        assert isinstance(self.model, BaseModel)
        logger.info(f'Build {self.model}')

    def put_batch_to_device(self, batch, device=None):
        if device is None:
            device = self.device

        if isinstance(batch, torch.Tensor):
            batch = batch.to(device)
            return batch
        elif isinstance(batch, dict):
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)
                elif isinstance(value, dict) or isinstance(value, list):
                    batch[key] = self.put_batch_to_device(value, device=device)
                # retain other value types in the batch dict
            return batch
        elif isinstance(batch, list):
            new_batch = []
            for value in batch:
                if isinstance(value, torch.Tensor):
                    new_batch.append(value.to(device))
                elif isinstance(value, dict) or isinstance(value, list):
                    new_batch.append(
                        self.put_batch_to_device(value, device=device))
                else:
                    # retain other value types in the batch list
                    new_batch.append(value)
            return new_batch
        else:
            raise Exception('Unsupported batch type {}'.format(type(batch)))

    def prepare_batch(self, batch):
        return self.put_batch_to_device(batch, self.device)

    def train(self):
        self.train_loaders = self.tab_data.train_loaders
        self.val_loaders = self.tab_data.val_loaders

        logger.info('Training starts')
        num_iters = self.config.sampler_per_epoch // self.config.batch_size

        self.models = []

        for fold_idx in range(len(self.train_loaders)):
            self.get_model()

            train_loader = self.train_loaders[fold_idx]
            val_loader = self.val_loaders[fold_idx]

            best_model = None
            best_val = np.inf

            for epoch in range(self.config.max_epochs):
                self.model.train()
                self.model.train_epoch_start()

                for iter, batch in enumerate(cycle(train_loader)):
                    if iter >= num_iters:
                        break

                    inputs, _, pseudo_labels = self.prepare_batch(batch)
                    # pseudo_labels = pseudo_labels.select(-1, -1).unsqueeze(dim=-1)
                    # pseudo_labels = pseudo_labels.mean(dim=-1, keepdim=True) + pseudo_labels.std(dim=-1, keepdim=True, unbiased=False)
                    
                    pseudo_labels = pseudo_labels.select(-1, -1).unsqueeze(dim=-1)
                    self.model.train_step(inputs, pseudo_labels, epoch)

                msg = self.model.train_epoch_end()
                val_loss = self.eval_on(val_loader, epoch)

                if val_loss < best_val:
                    best_val = val_loss
                    best_model = copy.deepcopy(self.model)

                msg = f'Epoch {epoch}, Train Message {msg}, Validation Loss {val_loss}'
                logger.info(msg)

            self.models.append(best_model)

    def eval_on(self, dataloader, epoch):
        assert isinstance(self.model, BaseModel)
        self.model.eval()

        running_loss = 0.0

        for batch in dataloader:
            inputs, _, pseudo_labels = self.prepare_batch(batch)
            pseudo_labels = pseudo_labels.select(-1, -1).unsqueeze(dim=-1)

            with torch.no_grad():
                loss = self.model.val_step(inputs, pseudo_labels, epoch)
                running_loss += loss.item()

        return running_loss / len(dataloader)

    def decision_function_mt(self, rep):
        scores = []
        labels = []
        means = []
        stds = []

        dataloader = self.tab_data.entire_loader

        for inputs, outputs, pseudo_labels in dataloader:
            inputs = self.prepare_batch(inputs)
            score = 0
            mean = 0
            std = 0
            for model in self.models:
                with torch.no_grad():
                    # score += model.decision_function(inputs, pseudo_labels, self.config.max_epochs)
                    score += model.decision_function(inputs, pseudo_labels, self.config.max_epochs)[0]
                    mean += model.decision_function(inputs, pseudo_labels, self.config.max_epochs)[1]
                    std += model.decision_function(inputs, pseudo_labels, self.config.max_epochs)[2]
            score /= len(self.models)
            mean /= len(self.models)
            std /= len(self.models)
            
            scores.append(self.put_batch_to_device(score, 'cpu'))
            labels.append(self.put_batch_to_device(outputs, 'cpu'))
            means.append(self.put_batch_to_device(mean, 'cpu'))
            stds.append(self.put_batch_to_device(std, 'cpu'))
        
        scores = torch.cat(scores).data.numpy().reshape(-1, 1)
        scaler = MinMaxScaler()
        scaler = scaler.fit(scores)
        # scores = MinMaxScaler().fit_transform(scores).reshape(-1, )
        
        scores = scores.reshape(-1, )
        # scores = scaler.transform(scores).reshape(-1, )
        
        
        means = torch.cat(means).data.numpy().reshape(-1, 1)
        # means = (means - scaler.data_min_) / (scaler.data_max_ - scaler.data_min_)
        
        stds = torch.cat(stds).data.numpy().reshape(-1, 1)
        # stds = (stds - scaler.data_min_) / (scaler.data_max_ - scaler.data_min_)
        
        #means = torch.cat(means).data.numpy().reshape(-1, 1)
        #stds = torch.cat(stds).data.numpy().reshape(-1, 1)
        print(scores[0], means[0], stds[0])
        

        np.save(f'./results/{self.config.data_path}_{self.config.target_mean_weight}_{self.config.target_std_weight}_mt_pseudo_label', self.tab_data.pseudo_labels)
        np.save(f'./results/{self.config.data_path}_{self.config.target_mean_weight}_{self.config.target_std_weight}_mt_score', scores)
        np.save(f'./results/{self.config.data_path}_{self.config.target_mean_weight}_{self.config.target_std_weight}_mt_label', self.tab_data.outputs)
        np.save(f'./results/{self.config.data_path}_{self.config.target_mean_weight}_{self.config.target_std_weight}_mt_mean_{rep}', means)
        np.save(f'./results/{self.config.data_path}_{self.config.target_mean_weight}_{self.config.target_std_weight}_mt_std_{rep}', stds)

        labels = torch.cat(labels).data.numpy().reshape(-1, )

        metrics = {
            'aucroc': roc_auc_score(labels, scores),
            'ap': average_precision_score(labels, scores)
        }

        logger.info(
            f'Performance on Dataset {self.config.data_path}, AUC ROC {metrics["aucroc"]}, AP {metrics["ap"]}'
        )

        return metrics, labels, scores
    
    def decision_function_co_train(self, rep):
        scores = []
        labels = []

        dataloader = self.tab_data.entire_loader

        for inputs, outputs, pseudo_labels in dataloader:
            inputs = self.prepare_batch(inputs)
            score = 0
            for model in self.models:
                with torch.no_grad():
                    score += model.decision_function(inputs, pseudo_labels, self.config.max_epochs)
            score /= len(self.models)

            scores.append(self.put_batch_to_device(score, 'cpu'))
            labels.append(self.put_batch_to_device(outputs, 'cpu'))

        scores = torch.cat(scores).data.numpy().reshape(-1, 1)
        scores = MinMaxScaler().fit_transform(scores).reshape(-1, )
        
        # labels is(is not) shuffle, which is depend on data loader(shuffle). self.tab_data.outputs is not shuffle
        labels = torch.cat(labels).data.numpy().reshape(-1, )
        
        # # save the best result among all reps
        # if roc_auc_score(labels, scores) > self.best_error:
        #     self.best_error = roc_auc_score(labels, scores)
        
        #     np.save(f'./Results/{self.config.data_path}_{self.config.model}_{self.config.pseudo_model}_{self.config.experiment_type}_pseudo_label_student', self.tab_data.pseudo_labels)
        #     np.save(f'./Results/{self.config.data_path}_{self.config.model}_{self.config.pseudo_model}_{self.config.experiment_type}_score_student', scores)
        #     np.save(f'./Results/{self.config.data_path}_{self.config.model}_{self.config.pseudo_model}_{self.config.experiment_type}_label_student', self.tab_data.outputs)

        # save all reps
        if roc_auc_score(labels, scores) > self.best_aucroc_error:
            self.best_aucroc_error = roc_auc_score(labels, scores)
            self.best_aucroc_rep = rep
            
        if average_precision_score(labels, scores) > self.best_ap_error:
            self.best_ap_error = average_precision_score(labels, scores)
            self.best_ap_rep = rep
    
        # np.save(f'./Results/{self.config.data_path}_{self.config.model}_{self.config.pseudo_model}_{self.config.experiment_type}_{rep}_pseudo_label_student', self.tab_data.pseudo_labels)
        # np.save(f'./Results/{self.config.data_path}_{self.config.model}_{self.config.pseudo_model}_{self.config.experiment_type}_{rep}_score_student', scores)
        # np.save(f'./Results/{self.config.data_path}_{self.config.model}_{self.config.pseudo_model}_{self.config.experiment_type}_{rep}_label_student', self.tab_data.outputs)
        np.save(f'./{read_file}/{self.config.data_path}_{self.config.model}_{self.config.pseudo_model}_{self.config.experiment_type}_{rep}_pseudo_label_student', self.tab_data.pseudo_labels)
        np.save(f'./{read_file}/{self.config.data_path}_{self.config.model}_{self.config.pseudo_model}_{self.config.experiment_type}_{rep}_score_student', scores)
        np.save(f'./{read_file}/{self.config.data_path}_{self.config.model}_{self.config.pseudo_model}_{self.config.experiment_type}_{rep}_label_student', self.tab_data.outputs)


        metrics = {
            'aucroc': roc_auc_score(labels, scores),
            'ap': average_precision_score(labels, scores)
        }

        logger.info(
            f'Performance on Dataset {self.config.data_path}, AUC ROC {metrics["aucroc"]}, AP {metrics["ap"]}'
        )

        return metrics, labels, scores

    def decision_function_co_train_baseline(self):
        scores = []
        labels = []

        dataloader = self.tab_data.entire_loader

        for inputs, outputs, pseudo_labels in dataloader:
            inputs = self.prepare_batch(inputs)
            score = 0
            for model in self.models:
                with torch.no_grad():
                    score += model.decision_function(inputs, pseudo_labels, self.config.max_epochs)
            score /= len(self.models)

            scores.append(self.put_batch_to_device(score, 'cpu'))
            labels.append(self.put_batch_to_device(outputs, 'cpu'))

        scores = torch.cat(scores).data.numpy().reshape(-1, 1)
        scores = MinMaxScaler().fit_transform(scores).reshape(-1, )
        
        # labels is(is not) shuffle, which is depend on data loader(shuffle). self.tab_data.outputs is not shuffle
        labels = torch.cat(labels).data.numpy().reshape(-1, )
        
        # np.save(f'./Results/{self.config.data_path}_{self.config.model}_{self.config.pseudo_model}_{self.config.experiment_type}_pseudo_label_student', self.tab_data.pseudo_labels)
        # np.save(f'./Results/{self.config.data_path}_{self.config.model}_{self.config.pseudo_model}_{self.config.experiment_type}_score_student', scores)
        # np.save(f'./Results/{self.config.data_path}_{self.config.model}_{self.config.pseudo_model}_{self.config.experiment_type}_label_student', self.tab_data.outputs)
        np.save(f'./{read_file}/{self.config.data_path}_{self.config.model}_{self.config.pseudo_model}_{self.config.experiment_type}_pseudo_label_student', self.tab_data.pseudo_labels)
        np.save(f'./{read_file}/{self.config.data_path}_{self.config.model}_{self.config.pseudo_model}_{self.config.experiment_type}_score_student', scores)
        np.save(f'./{read_file}/{self.config.data_path}_{self.config.model}_{self.config.pseudo_model}_{self.config.experiment_type}_label_student', self.tab_data.outputs)

        metrics = {
            'aucroc': roc_auc_score(labels, scores),
            'ap': average_precision_score(labels, scores)
        }

        logger.info(
            f'Performance on Dataset {self.config.data_path}, AUC ROC {metrics["aucroc"]}, AP {metrics["ap"]}'
        )

        return metrics, labels, scores
    
    def decision_function_co_train_baseline_cascade(self, rep):
        scores = []
        infer_scores = []
        labels = []

        dataloader = self.tab_data.entire_loader

        for inputs, outputs, pseudo_labels in dataloader:
            inputs = self.prepare_batch(inputs)
            score = 0
            infer_score = 0
            for model in self.models:
                with torch.no_grad():
                    score += model.decision_function(inputs, pseudo_labels, self.config.max_epochs)[0]
                    infer_score += model.decision_function(inputs, pseudo_labels, self.config.max_epochs)[1]
            score /= len(self.models)
            infer_score /= len(self.models)

            scores.append(self.put_batch_to_device(score, 'cpu'))
            infer_scores.append(self.put_batch_to_device(infer_score, 'cpu'))
            labels.append(self.put_batch_to_device(outputs, 'cpu'))

        scores = torch.cat(scores).data.numpy().reshape(-1, 1)
        scores = MinMaxScaler().fit_transform(scores).reshape(-1, )
        infer_scores = torch.cat(infer_scores).data.numpy().reshape(-1, 1)
        infer_scores = MinMaxScaler().fit_transform(infer_scores).reshape(-1, )
        
        # labels is(is not) shuffle, which is depend on data loader(shuffle). self.tab_data.outputs is not shuffle
        labels = torch.cat(labels).data.numpy().reshape(-1, )
    
        # np.save(f'./Results/{self.config.data_path}_{self.config.model}_{self.config.pseudo_model}_{self.config.experiment_type}_{rep}_pseudo_label_student', self.tab_data.pseudo_labels)
        # np.save(f'./Results/{self.config.data_path}_{self.config.model}_{self.config.pseudo_model}_{self.config.experiment_type}_{rep}_score_student', infer_scores)
        # np.save(f'./Results/{self.config.data_path}_{self.config.model}_{self.config.pseudo_model}_{self.config.experiment_type}_{rep}_label_student', self.tab_data.outputs)
        np.save(f'./{read_file}/{self.config.data_path}_{self.config.model}_{self.config.pseudo_model}_{self.config.experiment_type}_{rep}_pseudo_label_student', self.tab_data.pseudo_labels)
        np.save(f'./{read_file}/{self.config.data_path}_{self.config.model}_{self.config.pseudo_model}_{self.config.experiment_type}_{rep}_score_student', infer_scores)
        np.save(f'./{read_file}/{self.config.data_path}_{self.config.model}_{self.config.pseudo_model}_{self.config.experiment_type}_{rep}_label_student', self.tab_data.outputs)

        metrics = {
            'aucroc': roc_auc_score(labels, scores),
            'ap': average_precision_score(labels, scores)
        }

        logger.info(
            f'Performance on Dataset {self.config.data_path}, AUC ROC {metrics["aucroc"]}, AP {metrics["ap"]}'
        )

        return metrics, labels, scores
    
    def boost_train(self):
        # 10, 20 reps
        for rep in range(20):
            self.train()
            _, _, scores = self.decision_function_mt(rep)
            self.tab_data.update_pseudo_label(scores)
            self.tab_data.build_train_dataloaders()
            self.tab_data.build_entire_dataloader()

    def boost_co_train(self):
        self.best_aucroc_error = -1e10
        self.best_ap_error = -1e10
        if self.config.experiment_type == 'uadb':
            for rep in range(50):
                self.train()
                _, _, scores = self.decision_function_co_train(rep)
                self.tab_data.update_pseudo_label(scores)
                self.tab_data.build_train_dataloaders()
                self.tab_data.build_entire_dataloader()
                
        elif self.config.experiment_type == 'base_mean_cascade' or self.config.experiment_type == 'base_std_cascade':
            for rep in range(50):
                self.train()
                _, _, scores = self.decision_function_co_train_baseline_cascade(rep)
                self.tab_data.update_pseudo_label(scores)
                self.tab_data.build_train_dataloaders()
                self.tab_data.build_entire_dataloader()                   
        elif self.config.experiment_type == 'base_mean' or self.config.experiment_type == 'base_std':
            self.train()
            _, _, scores = self.decision_function_co_train_baseline()                        
