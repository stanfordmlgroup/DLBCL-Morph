import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from ignite.metrics import Accuracy

from models import get_model
from eval import get_loss_fn
from data import ImageClassificationDataset
from .logger import TFLogger


class Model(pl.LightningModule, TFLogger):
    """Standard interface for the trainer to interact with the model."""

    def __init__(self, args):
        super(Model, self).__init__()
        self.hparams = args
        self.model = get_model(args)
        self.loss = get_loss_fn(args)
        self.val_acc = Accuracy()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        """
        Returns:
            A dictionary of loss and metrics, with:
                loss(required): loss used to calculate the gradient
                log: metrics to be logged to the TensorBoard and metrics.csv
                progress_bar: metrics to be logged to the progress bar
                              and metrics.csv
        """
        x, y = batch
        logits = self.forward(x)
        loss = self.loss(logits.view(-1), y)
        return {'loss': loss, 'log': {'train_loss': loss}}

    def validation_step(self, batch, batch_nb):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss(logits.view(-1), y)
        y_hat = (logits > 0).float()
        print(logits, y_hat, y)
        self.val_acc.update((y_hat, y))
        return {'val_loss': loss}

    def validation_end(self, outputs):
        """
        Aggregate and return the validation metrics

        Args:
        outputs: A list of dictionaries of metrics from `validation_step()'
        Returns: None
        Returns:
            A dictionary of loss and metrics, with:
                val_loss (required): validation_loss
                log: metrics to be logged to the TensorBoard and metrics.csv
                progress_bar: metrics to be logged to the progress bar
                              and metrics.csv
        """

        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = self.val_acc.compute()
        self.val_acc.reset()
        return {'val_loss': avg_loss,
                'log': {'avg_val_loss': avg_loss},
                'progress_bar':
                    {'avg_val_loss': avg_loss,
                     'avg_val_acc': avg_acc}}

    def test_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat.view(-1), y)
        return {'test_loss': loss, 'log': {'test_loss': loss}}

    def test_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        return {'avg_test_loss': avg_loss}

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=0.02)]

    @pl.data_loader
    def train_dataloader(self):
        dataset = ImageClassificationDataset(
            image_path=["images/test_image.png", "images/test_image.png"],
            labels=[0, 1],
            transforms=T.Compose([T.Resize((224, 224)), T.ToTensor()]))
        return DataLoader(dataset, shuffle=True,
                          batch_size=2, num_workers=8)

    @pl.data_loader
    def val_dataloader(self):
        dataset = ImageClassificationDataset(
            image_path=["images/test_image.png", "images/test_image.png"],
            labels=[0, 1],
            transforms=T.Compose([T.Resize((224, 224)), T.ToTensor()]))
        return DataLoader(dataset, shuffle=False,
                          batch_size=1, num_workers=8)

    @pl.data_loader
    def test_dataloader(self):
        dataset = ImageClassificationDataset(
            image_path=["images/test_image.png", "images/test_image.png"],
            labels=[0, 1],
            transforms=T.Compose([T.Resize((224, 224)), T.ToTensor()]))
        return DataLoader(dataset, shuffle=False,
                          batch_size=1, num_workers=8)
