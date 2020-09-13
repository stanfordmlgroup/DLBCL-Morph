import os
import fire
from pytorch_lightning import Trainer

from lightning import Model
from util import init_exp_folder, Args
from lightning import get_ckpt_callback, get_early_stop_callback
from lightning import get_logger


def train(save_dir="./sandbox",
          exp_name="DemoExperiment",
          model="ResNet18",
          gpus=1,
          pretrained=True,
          num_classes=1,
          log_save_interval=1,
          distributed_backend="dp",
          gradient_clip_val=0.5,
          max_epochs=1,
          patience=10,
          train_percent_check=1.0,
          tb_path="./sandbox/tb",
          loss_fn="BCE",
          weights_summary=None,
          ):
    """
    Run the training experiment.

    Args:
        save_dir: Path to save the checkpoints and logs
        exp_name: Name of the experiment
        model: Model name
        gpus: int. (ie: 2 gpus)
             OR list to specify which GPUs [0, 1] OR '0,1'
             OR '-1' / -1 to use all available gpus
        pretrained: Whether or not to use the pretrained model
        num_classes: Number of classes
        log_save_interval: Logging saving frequency (in batch)
        distributed_backend: Distributed computing mode
        gradient_clip_val:  Clip value of gradient norm
        train_percent_check: Proportion of training data to use
        max_epochs: Max number of epochs
                patience: number of epochs with no improvement after
                                  which training will be stopped.
        tb_path: Path to global tb folder
        loss_fn: Loss function to use
        weights_summary: Prints a summary of the weights when training begins.

    Returns: None

    """
    args = Args(locals())
    init_exp_folder(args)
    m = Model(args)
    trainer = Trainer(distributed_backend=distributed_backend,
                      gpus=gpus,
                      logger=get_logger(save_dir, exp_name),
                      checkpoint_callback=get_ckpt_callback(save_dir,
                                                            exp_name),
                      early_stop_callback=get_early_stop_callback(),
                      default_save_path=os.path.join(save_dir, exp_name),
                      log_save_interval=log_save_interval,
                      gradient_clip_val=gradient_clip_val,
                      train_percent_check=train_percent_check,
                      weights_summary=weights_summary,
                      max_epochs=max_epochs)
    trainer.fit(m)


def test(save_dir="./sandbox/",
         gpus=4,
         checkpoint_path="./sandbox/DemoExperiment/ckpts/_ckpt_epoch_0.ckpt"):
    """
    Run the testing experiment.

    Args:
        model: Model name
        save_dir: Path to save the checkpoints and logs
        gpus: int. (ie: 2 gpus)
             OR list to specify which GPUs [0, 1] OR '0,1'
             OR '-1' / -1 to use all available gpus
        checkpoint_path: Path for the experiment to load
        loss_fn: Loss function to use
    Returns: None

    """
    args = Args(locals())
    m = Model.load_from_checkpoint(checkpoint_path)
    trainer = Trainer(default_save_path=os.path.join(save_dir,
                                                     "result/test"),
                      gpus=gpus)
    trainer.test(m)


if __name__ == "__main__":
    fire.Fire()
