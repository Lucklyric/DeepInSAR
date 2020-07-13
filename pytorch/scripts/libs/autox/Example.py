import argparse
import numpy as np
import torch

from .logger import AutoLogger
from .model import SampleModel 
from .runner import AutoRunner


class DummyDB(object):
    def __init__(self, length):
        self.length = length
        self.data = torch.randn([10,10])
    def __len__(self):
        return self.length
    def __iter__(self):
        for i in range(self.length):
            yield self.data

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser = AutoRunner.add_args(parser)
    parser = SampleModel.add_args(parser)
    args = parser.parse_args()

    device = torch.device('cuda:0')
    model = SampleModel(args, device)

    runner = AutoRunner(**vars(args))
    
    runner.fit(model, DummyDB(300))

