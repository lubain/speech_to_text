import os
import ast
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
from argparse import ArgumentParser
from model import SpeechRecognition
from dataset import Data, collate_fn_padd
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

class SpeechModule(LightningModule):

    def __init__(self, model, args):
        super(SpeechModule, self).__init__()
        self.model = model
        self.criterion = nn.CTCLoss(blank=28, zero_infinity=True)
        self.args = args
        self.validation_outputs = []

    def forward(self, x, hidden):
        return self.model(x, hidden)

    def configure_optimizers(self):
        self.optimizer = optim.AdamW(self.model.parameters(), self.args['learning_rate'])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min',
            factor=0.50, patience=6
        )
        return {
            'optimizer': self.optimizer,
            'lr_scheduler': {
                'scheduler': self.scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }

    def step(self, batch):
        spectrograms, labels, input_lengths, label_lengths = batch
        bs = spectrograms.shape[0]
        hidden = self.model._init_hidden(bs)
        hn, c0 = hidden[0].to(self.device), hidden[1].to(self.device)
        output, _ = self(spectrograms, (hn, c0))
        output = F.log_softmax(output, dim=2)
        loss = self.criterion(output, labels, input_lengths, label_lengths)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        logs = {'loss': loss, 'lr': self.optimizer.param_groups[0]['lr']}
        self.log('train_loss', loss)
        return loss

    def train_dataloader(self):
        d_params = Data.parameters
        d_params.update(self.args['dparams_override'])
        train_dataset = Data(json_path=self.args['train_file'], **d_params)
        return DataLoader(dataset=train_dataset,
                          batch_size=self.args['batch_size'],
                          num_workers=self.args['data_workers'],
                          pin_memory=True,
                          collate_fn=collate_fn_padd)

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.validation_outputs.append(loss)
        self.log('val_loss', loss)
        return loss

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.validation_outputs).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        self.log('val_loss', avg_loss)
        self.validation_outputs.clear()

    def val_dataloader(self):
        d_params = Data.parameters
        d_params.update(self.args['dparams_override'])
        test_dataset = Data(json_path=self.args['valid_file'], **d_params, valid=True)
        return DataLoader(dataset=test_dataset,
                          batch_size=self.args['batch_size'],
                          num_workers=self.args['data_workers'],
                          collate_fn=collate_fn_padd,
                          pin_memory=True)


def checkpoint_callback(args):
    return ModelCheckpoint(
        dirpath=os.path.dirname(args['save_model_path']),
        filename=os.path.basename(args['save_model_path']),
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )

def main(args):
    h_params = SpeechRecognition.hyper_parameters
    h_params.update(args['hparams_override'])
    model = SpeechRecognition(**h_params)

    if args['load_model_from']:
        speech_module = SpeechModule.load_from_checkpoint(args['load_model_from'], model=model, args=args)
    else:
        speech_module = SpeechModule(model, args)

    logger = TensorBoardLogger(args['logdir'], name='speech_recognition')

    trainer = Trainer(
        max_epochs=args['epochs'],
        devices=args['devices'],
        accelerator=args['accelerator'],
        num_nodes=args['nodes'],
        logger=logger,
        gradient_clip_val=1.0,
        check_val_every_n_epoch=1,  # Validation à chaque époque
        callbacks=[checkpoint_callback(args)],
        log_every_n_steps=10  # Intervalle de journalisation plus petit
    )
    trainer.fit(speech_module, ckpt_path=args['resume_from_checkpoint'] if args['resume_from_checkpoint'] else None)

# Définir les arguments directement
args = {
    'nodes': 1,
    'devices': 1,
    'accelerator': "cpu",
    'data_workers': 0,
    'dist_backend': 'ddp',
    'train_file': 'output/train.json',
    'valid_file': 'output/test.json',
    'valid_every': 1000,  # Vous pouvez supprimer cette ligne ou la définir de manière appropriée
    'save_model_path': 'model/model.ckpt',
    'load_model_from': None,
    'resume_from_checkpoint': None,
    'logdir': 'tb_logs',
    'epochs': 10,
    'batch_size': 64,
    'learning_rate': 1e-3,
    'pct_start': 0.3,
    'div_factor': 100,
    'hparams_override': {},
    'dparams_override': {}
}

# Exécuter la fonction principale
main(args)