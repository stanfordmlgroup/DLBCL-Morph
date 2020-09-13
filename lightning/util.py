"""Define Logger class for logging information to stdout and disk."""
import json
import os
from os.path import join
from pytorch_lightning.logging import TestTubeLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


def get_ckpt_dir(save_path, exp_name):
    return os.path.join(save_path, exp_name, "ckpts")


def get_ckpt_callback(save_path, exp_name):
    ckpt_dir = os.path.join(save_path, exp_name, "ckpts")
    return ModelCheckpoint(filepath=ckpt_dir,
                           save_top_k=1,
                           verbose=True,
                           monitor='val_loss',
                           mode='min',
                           prefix='')


def get_early_stop_callback(patience=10):
    return EarlyStopping(monitor='val_loss',
                         patience=patience,
                         verbose=True,
                         mode='min')


def get_logger(save_path, exp_name):
    exp_dir = os.path.join(save_path, exp_name)
    return TestTubeLogger(save_dir=exp_dir,
                          name='lightning_logs',
                          version="0")
