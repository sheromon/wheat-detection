
import pytorch_lightning as pl
import torch
from torchvision.models.detection import faster_rcnn


class WheatModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self._loss_names = ('loss_classifier', 'loss_box_reg', 'loss_objectness', 'loss_rpn_box_reg')
        self.config = config
        # num_classes includes background, so wheat and background are two classes
        self.model = faster_rcnn.fasterrcnn_resnet50_fpn(num_classes=2)

    def forward(self, example):
        image, targets = example
        # pytorch lightning wants the loss calculate to be in the training step,
        # not in the forward method, but the torchvision models don't cooperate,
        # so here we are returning losses in forward
        loss_dict = self.model(image, targets)
        return loss_dict

    def training_step(self, batch, batch_idx):
        loss_dict = self(batch)
        total_loss = sum([loss_dict[key] for key in self._loss_names])
        return total_loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), 
            lr=self.config['train']['optimizer']['initial_lr'],
        )
