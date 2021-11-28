"""Evaluation script"""
import argparse
from pathlib import Path

import pytorch_lightning as pl

from wheat.config import load_config
from wheat.data_module import WheatDataModule
from wheat.model import WheatModel

# TODO: use ResourceManager API as recommended in
# https://setuptools.pypa.io/en/latest/userguide/datafiles.html
BASE_DIR = Path(__file__).parent.parent


def evaluate(config, args_dict, ckpt_path):
    """Run evaluation on the val set.

    :param config: configobj mapping of config paramters to values
    :param args_dict: dict of options for initializing PyTorch Lightning Trainer
    :param ckpt_path: path to checkpoint to load for inference
    """
    wheat_data_module = WheatDataModule(config)
    model = WheatModel(config)
    trainer = pl.Trainer(**args_dict)
    trainer.validate(model, wheat_data_module, ckpt_path=ckpt_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt_path', type=str,
                        help='Path to model checkpoint file')
    parser.add_argument('--config-path', type=str,
                        default=BASE_DIR/'config'/'config.ini',
                        help='Path to config file')
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    args_dict = vars(args)
    config = load_config(args_dict.pop('config_path'))
    ckpt_path = args_dict.pop('ckpt_path')
    evaluate(config, args_dict, ckpt_path)


if __name__ == '__main__':
    main()
