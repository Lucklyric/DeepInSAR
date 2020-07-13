import numpy as np
import torch
import typing
from abc import abstractmethod

class AutoModel(object):

    def __init__(self, *args, **kwargs):

        #: The current number of epoch
        self.current_epoch = 0

        #: The current global iteraction step
        self.global_step = 0

        #: A dictionary of modules {'feature_extractor': module}
        self.modules = {}

        #: A ref to the logger object
        self.logger = None

        #: A ref to the runner object
        self.runner = None

        #: Device information
        self.device = None
    
    def set_logger(self, logger):
        self.logger = logger

    def set_runner(self, runner):
        self.runner = runner
    
    def train(self):
        for module_name in self.modules:
            self.modules[module_name].train()

    def eval(self):
        for module_name in self.modules:
            self.modules[module_name].eval()
    
    def to(self, device):
        self.device = device
        for module_name in self.modules:
            self.modules[module_name] = self.modules[module_name].to(device)

    @staticmethod
    def add_args(parser):
        return parser

    @abstractmethod
    def forawrd(self, x):
        return NotImplementedError

    @abstractmethod
    def fit_step(self, batch):
        return NotImplementedError

    def on_save_checkpoint(self, ckpt_dir):
        pass

    def on_load_checkpoint(self, ckpt_dir):
        pass

    def on_batch_begin(self):
        pass

    def on_batch_end(self):
        pass

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self):
        pass

