from libs.model.deepinsar import DeepInSAR
from libs.data.dataloader import SimInSARDB
from libs.autox.runner.AutoRunner import AutoRunner
import torch
import argparse
from torch.utils.data import Dataset, DataLoader

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser()

    parser = AutoRunner.add_args(parser)
    parser = DeepInSAR.add_args(parser)

    device = torch.device('cuda:0')

    parser.add_argument('--sim_root', default='./sim', type=str)

    hparams = parser.parse_args()
    print(hparams)
    db = SimInSARDB(128, hparams.sim_root, int(1.6e5)*32, 1000)
    dataloader = DataLoader(db, batch_size=32, shuffle=True, num_workers=8)

    model = DeepInSAR(hparams, device)

    runner = AutoRunner(**(vars(hparams)))

    runner.fit(model, dataloader)

