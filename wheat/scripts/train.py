import argparse

import pytorch_lightning as pl

from wheat.config import load_config
from wheat.data_module import WheatDataModule
from wheat.model import WheatModel


def train(config, args):
    wheat_data_module = WheatDataModule(config)
    model = WheatModel(config)
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, wheat_data_module)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=str,
                        default='wheat/config/config.ini',
                        help='Path to config file')
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    config = load_config(args.config_path)
    train(config, args)


if __name__ == '__main__':
    main()
