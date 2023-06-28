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
        
        self.train_accuracy = Accuracy(task='binary', threshold=0.5)
        self.valid_accuracy = Accuracy(task='binary', threshold=0.5)

        self.train_precision = Precision(task='binary', threshold=0.5)
        self.valid_precision = Precision(task='binary', threshold=0.5)

        self.train_recall = Recall(task='binary', threshold=0.5)
        self.valid_recall = Recall(task='binary', threshold=0.5)
        
        self.training_step_outputs = []
        self.validation_step_outputs = [] 


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

        self.training_step_outputs.append({
            "loss": loss,
            "acc": acc,
            "prec": prec,
            "rec": rec
        })
        return loss
                                

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
        
        self.validation_step_outputs.append({
            "loss": loss,
            "acc": acc,
            "prec": prec,
            "rec": rec
        })
        
        return loss



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



    def on_train_epoch_end(self):

        acc, prec, rec, loss = (
            torch.stack([out['acc'] for out in self.training_step_outputs]),
            torch.stack([out['prec'] for out in self.training_step_outputs]),
            torch.stack([out['rec'] for out in self.training_step_outputs]),
            torch.stack([out['loss'] for out in self.training_step_outputs])
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
        
        self.training_step_outputs.clear()


    def on_validation_epoch_end(self) -> None:
    

        acc, prec, rec, loss = (
            torch.stack([out['acc'] for out in self.validation_step_outputs]),
            torch.stack([out['prec'] for out in self.validation_step_outputs]),
            torch.stack([out['rec'] for out in self.validation_step_outputs]),
            torch.stack([out['loss'] for out in self.validation_step_outputs])
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


        self.validation_step_outputs.clear()
        
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