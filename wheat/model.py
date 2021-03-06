"""PyTorch Lightning Module with torchvision object detection model"""
import os
from pathlib import Path

import brambox
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torchvision.models.detection import faster_rcnn


class WheatModel(pl.LightningModule):
    """PyTorch Lightning Module with torchvision object detection model"""

    def __init__(self, config):
        super().__init__()
        self._loss_names = (
            'loss_classifier', 'loss_box_reg',
            'loss_objectness', 'loss_rpn_box_reg',
        )
        self.config = config
        self.learning_rate = self.config['train']['optimizer']['initial_lr']
        # num_classes includes background, so wheat and background are two classes
        self.model = faster_rcnn.fasterrcnn_resnet50_fpn(num_classes=2)

    def forward(self, example):
        image, targets = example
        # pytorch lightning wants the loss calculate to be in the training step,
        # not in the forward method, but the torchvision models don't cooperate,
        # so here we are returning losses in forward. note that in eval mode, we
        # will be returning a list of predictions instead of losses.
        return self.model(image, targets)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.learning_rate,
            momentum=0.8,
        )
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=self.config['train']['optimizer']['initial_lr'],
            max_lr=self.config['train']['optimizer']['max_lr'],
        )
        lr_scheduler_config = {
            'scheduler': lr_scheduler,
            'interval': 'step',
            'frequency': 1,
        }
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        """Run the model in training mode.

        :return: scalar tensor with total loss value
        """
        # note from pytorch docs: Prior to PyTorch 1.1.0, the learning rate
        # scheduler was expected to be called before the optimizer???s update;
        # #1.1.0 changed this behavior in a BC-breaking way.
        loss_dict = self(batch)
        total_loss = sum([loss_dict[key] for key in self._loss_names])
        loss_dict['total_loss'] = total_loss
        self.log_dict(loss_dict, on_step=True, batch_size=len(batch[1]))
        return total_loss

    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        """Run the model in eval mode.

        :return: tuple (labels, predictions) where labels and predictions are
            lists with length equal to the batch size; each element of labels is
            a dict with keys 'boxes' and 'labels', and eachelement of predictions
            is a dict with keys 'boxes', 'labels', and 'scores'
        """
        # torchvision generalized r-cnn models return different outputs when
        # running in train mode versus eval mode. I wish the forward method just
        # returned raw predictions instead of the full process of calculating
        # loss or creating predicted bounding boxes, but hey, it's free code,
        # and beggars can't be choosers.
        predictions = self(batch)
        # batch[0] is a list of input images, and we don't need those for eval
        return batch[1], predictions

    def validation_epoch_end(self, outputs):
        """Calculate average precision (AP) and log it."""
        det_df, anno_df = self.compile_eval_dataframes(outputs)
        # if these EVs are set, write the dataframes to disk
        # this lets us do offline analysis later on
        save_dir = os.environ.get('CMD_WHEAT_OUTPUT_DIR')
        print('save_dir:', save_dir)
        if save_dir:
            det_save_path = Path(save_dir)/'det.csv'
            anno_save_path = Path(save_dir)/'anno.csv'
            det_df.to_csv(det_save_path)
            anno_df.to_csv(anno_save_path)
        # kaggle global wheat detection evaluation metric is AP calculated from
        # 0.5 to 0.75 IOU in 0.05 increments. I don't want to calculate AP at
        # every value, so just calculate it at 0.5 and 0.75.
        for iou_threshold in [0.5, 0.75]:
            df_pr = brambox.stat.pr(det_df, anno_df, threshold=iou_threshold)
            ap = brambox.stat.ap(df_pr)
            metric_name = 'ap' + str(round(100*iou_threshold))
            self.log(metric_name, ap, on_epoch=True)

    def compile_eval_dataframes(self, outputs):
        """Reformat eval outputs into detection and annotation dataframes.

        This output format is consistent with the required input format for the
        brambox package.

        :return: tuple of pandas dataframes (det_df, anno_df), where each dataframe
            has columns that define the ground truth or predicted bounding box
            'x_top_left', 'y_top_left', 'width', 'height'; an image ID 'image';
            a class label 'class_label', and for detections, a score 'confidence'
        """
        anno_df_list = []
        det_df_list = []
        for ibatch, val_outputs_batch in enumerate(outputs):
            for isample, (anno, pred) in enumerate(zip(*val_outputs_batch)):
                ind = ibatch * len(val_outputs_batch[0]) + isample
                anno_data = {
                    'image': ind * np.ones(len(anno['labels']), dtype=np.int64),
                    'x_top_left': anno['boxes'][:, 0].cpu().numpy().astype(np.float64),
                    'y_top_left': anno['boxes'][:, 1].cpu().numpy().astype(np.float64),
                    'width': (anno['boxes'][:, 2] - anno['boxes'][:, 0]).cpu() \
                        .numpy().astype(np.float64),
                    'height': (anno['boxes'][:, 3] - anno['boxes'][:, 1]).cpu() \
                        .numpy().astype(np.float64),
                    'class_index': anno['labels'].cpu().numpy().astype(np.int64),
                }
                anno_df = pd.DataFrame(data=anno_data)
                anno_df['class_label'] = anno_df['class_index']
                anno_df['ignore'] = False
                anno_df_list.append(anno_df)

                pred_data = {
                    'image': ind * np.ones(len(pred['labels']), dtype=np.int64),
                    'x_top_left': pred['boxes'][:, 0].cpu().numpy().astype(np.float64),
                    'y_top_left': pred['boxes'][:, 1].cpu().numpy().astype(np.float64),
                    'width': (pred['boxes'][:, 2] - pred['boxes'][:, 0]).cpu() \
                        .numpy().astype(np.float64),
                    'height': (pred['boxes'][:, 3] - pred['boxes'][:, 1]).cpu() \
                        .numpy().astype(np.float64),
                    'class_index': pred['labels'].cpu().numpy().astype(np.int64),
                    'confidence': pred['scores'].cpu().numpy().astype(np.float64),
                }
                det_df = pd.DataFrame(data=pred_data)
                det_df['class_label'] = det_df['class_index']
                det_df_list.append(det_df)
        return pd.concat(det_df_list), pd.concat(anno_df_list)
