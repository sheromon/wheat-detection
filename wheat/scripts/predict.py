"""Prediction script"""
import argparse

import pytorch_lightning as pl

from wheat.config import load_config
from wheat.data_module import WheatDataModule
from wheat.dataset import WheatDataset
from wheat.model import WheatModel


def predict(config, args_dict: dict, ckpt_path: str):
    """Run inference on the test set and save predictions.

    :param config: configobj mapping of config paramters to values
    :param args_dict: dict of options for initializing PyTorch Lightning Trainer
    :param ckpt_path: path to checkpoint to load for inference
    """
    wheat_data_module = WheatDataModule(config)
    model = WheatModel(config)
    trainer = pl.Trainer(**args_dict)
    results = trainer.predict(model, wheat_data_module, ckpt_path=ckpt_path)
    score_threshold = config['predict']['score_threshold']
    save_submission(results, wheat_data_module.test_dataset, score_threshold)


def save_submission(results, test_dataset: WheatDataset, score_threshold: float,
                    save_path='submission.csv'):
    """Save inference outputs in format specified by Kaggle competition.

    :param results: outputs from Trainer.predict; list of list of dicts with keys
        'scores', 'boxes', and 'labels'
    :param score_threshold: predictions with scores below this value will be discarded
    """
    image_ids = test_dataset.image_ids
    header = 'image_id,PredictionString\n'
    with open(save_path, 'w', encoding='utf-8') as file_obj:
        file_obj.write(header)
        for image_id, predictions in zip(image_ids, results):
            keep_inds = predictions[0]['scores'] > score_threshold
            keep_scores = (predictions[0]['scores'][keep_inds]).cpu().numpy()
            keep_boxes = predictions[0]['boxes'][keep_inds, :].cpu().numpy()
            # convert last two columns to width and height instead of xmax, ymax
            keep_boxes[:, 2:] -= keep_boxes[:, :2]
            line = f'{image_id},'
            for ind, score in enumerate(keep_scores):
                line += f' {score} '
                line += ' '.join([str(round(val)) for val in keep_boxes[ind]])
            line += '\n'
            file_obj.write(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt_path', type=str,
                        help='Path to model checkpoint file')
    parser.add_argument('--config-path', type=str, default=None,
                        help='Path to config file')
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    args_dict = vars(args)
    config = load_config(args_dict.pop('config_path'))
    ckpt_path = args_dict.pop('ckpt_path')
    predict(config, args_dict, ckpt_path)


if __name__ == '__main__':
    main()
