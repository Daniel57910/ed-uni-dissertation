import pdb

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy
from torchmetrics import Precision
from torchmetrics import Recall
METADATA_INDEX = 14
PLATFORM_TIME_INDEX = 4
USER_ID_INDEX = 1

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
        
    

        metadata, features  = self._extract_features(batch)
        y = metadata[:, 0].unsqueeze(1)

        if self.model_name.startswith('ordinal'):
            y_hat = self(features)
            
        if self.model_name.startswith('heuristic'):
            cum_platform_time = metadata[:, PLATFORM_TIME_INDEX]
            y_hat = torch.where(cum_platform_time <= 25, torch.zeros_like(y), self(features))
            
        if self.model_name.startswith('embedded'):
            user_id = metadata[:, USER_ID_INDEX]
            concatenated = torch.cat((user_id.unsqueeze(1), features), dim=2)
            y_hat = self(concatenated)
            
            
        
        # if 'embedded' in self.model_name:
        #     concatenated = torch.cat((ordinal_features, categorical_features), dim=2)
        #     assert concatenated.shape == (batch.shape[0], self.n_sequences, 19), 'concatenated shape is wrong'
        #     y_hat = self(torch.cat((ordinal_features, categorical_features), dim=2))


        loss = self.loss(y_hat, y)

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
        
        metadata, features = tensor[:, :METADATA_INDEX], tensor[:, METADATA_INDEX:] 
                
        features = torch.flip(
            torch.reshape(features, (features.shape[0], self.n_sequences, self.n_features)),
            dims=[1]
        )
        
        return metadata, features



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