import pdb


import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy
from torchmetrics import Precision
from torchmetrics import Recall
ZERO_HEURISTIC_RATE = 10
ORDINAL_FEATURE_INDEX = 17

import pdb

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy
from torchmetrics import Precision
from torchmetrics import Recall
ZERO_HEURISTIC_RATE = 10
ORDINAL_FEATURE_INDEX = 17



class ModelBase(LightningModule):

    def __init__(self) -> None:
        super().__init__()

        self.loss = nn.BCEWithLogitsLoss()

        self.runtime_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.train_accuracy = Accuracy(task='binary', threshold=0.5)
        self.valid_accuracy = Accuracy(task='binary', threshold=0.5)

        self.train_precision = Precision(task='binary', threshold=0.5)
        self.valid_precision = Precision(task='binary', threshold=0.5)

        self.train_recall = Recall(task='binary', threshold=0.5)
        self.valid_recall = Recall(task='binary', threshold=0.5)

        self = self.to(self.runtime_device)

    def training_step(self, batch, batch_idx):
        loss, acc, prec, rec = self._run_step(batch, 'train')

        self.log(
            'loss/train',
            loss,
            logger=True,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
        )

        self.log(
            'loss_train',
            loss,
            logger=False,
            prog_bar=False,
            on_step=True,
            on_epoch=True,
            sync_dist=True
        )

        return {
            "loss": loss,
            "acc": acc,
            "prec": prec,
            "rec": rec
        }


    def validation_step(self, batch, batch_idx):
        loss, acc, prec, rec = self._run_step(batch, 'valid')

        self.log(
            'loss/valid',
            loss,
            logger=True,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
        )

        self.log(
            'loss_valid',
            loss,
            logger=False,
            prog_bar=False,
            on_step=True,
            on_epoch=True,
            sync_dist=True
        )

        return {
            "loss": loss,
            "acc": acc,
            "prec": prec,
            "rec": rec
        }


    def _run_step(self, batch, type):

        y, total_events, categorical_features, ordinal_features  = self._extract_features(batch)

        if 'ordinal' in self.model_name:
            y_hat = self(ordinal_features)
        
        if 'embedded' in self.model_name:
            concatenated = torch.cat((ordinal_features, categorical_features), dim=2)
            assert concatenated.shape == (batch.shape[0], self.n_sequences, 19), 'concatenated shape is wrong'
            y_hat = self(torch.cat((ordinal_features, categorical_features), dim=2))

        if self.zero_heuristic:
            y_hat = torch.where(total_events <= ZERO_HEURISTIC_RATE, torch.zeros_like(y_hat), y_hat)

        loss = self.loss(y_hat, y)
        y = y.int()

        if 'train' in type:
            acc = self.train_accuracy(y_hat, y)
            prec = self.train_precision(y_hat, y)
            rec = self.train_recall(y_hat, y)

        else:
            acc = self.valid_accuracy(y_hat, y)
            prec = self.valid_precision(y_hat, y)
            rec = self.valid_recall(y_hat, y)

        return loss, acc, prec, rec

    def _extract_features(self, tensor):

        label, total_events, user_id, project_id, features, shifters = (
            tensor[:, 0], tensor[:, 1],  tensor[:, 2],
            tensor[:, 3], tensor[:, 5:5+17], tensor[:, 5+17:]
        )

        shifters = torch.reshape(shifters, (shifters.shape[0], self.n_sequences-1, 18))
        shifter_project_id, shifter_features = shifters[:, :, 0], shifters[:, :, 1:]

        project_id = torch.flip(torch.cat((project_id.unsqueeze(1), shifter_project_id), dim=1), dims=[1]).long()
        features = torch.flip(torch.cat((features.unsqueeze(1), shifter_features), dim=1), dims=[1])

        user_id = user_id.unsqueeze(1).repeat(1, self.n_sequences).long()
        user_id = torch.where(project_id == 0, 0, user_id)

        user_id, project_id = user_id.unsqueeze(2), project_id.unsqueeze(2)

        user_project_concat = torch.cat((user_id, project_id), dim=2)

        assert user_project_concat.shape == (user_id.shape[0], self.n_sequences, 2), 'user_project_concat shape is wrong'
        return (
            label.unsqueeze(1), 
            total_events.unsqueeze(1), 
            user_project_concat,
            features
        )

    def training_epoch_end(self, outputs):

        acc, prec, rec, loss = (
            torch.stack([out['acc'] for out in outputs]),
            torch.stack([out['prec'] for out in outputs]),
            torch.stack([out['rec'] for out in outputs]),
            torch.stack([out['loss'] for out in outputs])
        )

        acc, prec, rec, loss = (
            torch.mean(acc),
            torch.mean(prec),
            torch.mean(rec),
            torch.mean(loss)
        )

        self.logger.experiment.add_scalar('acc/train', acc, self.current_epoch)
        self.logger.experiment.add_scalar('prec/train', prec, self.current_epoch)
        self.logger.experiment.add_scalar('rec/train', rec, self.current_epoch)
        self.logger.experiment.add_scalar('loss_e/train', loss, self.current_epoch)


    def validation_epoch_end(self, outputs):

        acc, prec, rec, loss = (
            torch.stack([out['acc'] for out in outputs]),
            torch.stack([out['prec'] for out in outputs]),
            torch.stack([out['rec'] for out in outputs]),
            torch.stack([out['loss'] for out in outputs])
        )

        acc, prec, rec, loss = (
            torch.mean(acc),
            torch.mean(prec),
            torch.mean(rec),
            torch.mean(loss)
        )

        self.logger.experiment.add_scalar('acc/valid', acc, self.current_epoch)
        self.logger.experiment.add_scalar('prec/valid', prec, self.current_epoch)
        self.logger.experiment.add_scalar('rec/valid', rec, self.current_epoch)
        self.logger.experiment.add_scalar('loss_e/valid', loss, self.current_epoch)

    def configure_optimizers(self):
        # equation for adam optimizer
        """
        m_t = beta_1 * m_{t-1} + (1 - beta_1) * g_t
        v_t = beta_2 * v_{t-1} + (1 - beta_2) * g_t^2
        m_cap = m_t / (1 - beta_1^t)
        v_cap = v_t / (1 - beta_2^t)
        w_t = w_{t-1} - lr * m_cap / (sqrt(v_cap) + eps)
        """
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)