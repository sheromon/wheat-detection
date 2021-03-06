"""Training script"""
import argparse

import pytorch_lightning as pl

from wheat.config import load_config
from wheat.data_module import WheatDataModule
from wheat.model import WheatModel


def train(config, args_dict):
    wheat_data_module = WheatDataModule(config)
    model = WheatModel(config)

    checkpointer = pl.callbacks.ModelCheckpoint(monitor='ap75', mode='max')
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch', log_momentum=True)
    trainer = pl.Trainer(**args_dict, callbacks=[checkpointer, lr_monitor])
    trainer.fit(model, wheat_data_module)
    trainer.validate(model, wheat_data_module)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=str, default=None,
                        help='Path to config file')
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    args_dict = vars(args)
    config = load_config(args_dict.pop('config_path'))
    train(config, args_dict)


if __name__ == '__main__':
    main()
