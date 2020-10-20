from ..autox.model import AutoModel
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse


class BasicBlock(nn.Module):

    def __init__(self, in_planes, out_planes=24, g_rate=3):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes * g_rate, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes * g_rate)
        self.conv2 = nn.Conv2d(out_planes * g_rate, out_planes, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        out = self.conv2(self.bn2(self.conv1(torch.relu(self.bn1(x)))))
        out = torch.relu(out)
        return torch.cat([x, out], 1)


class DenseBlock(nn.Module):

    def __init__(self, nb_layers, in_planes, growth_rate, block):
        super(DenseBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, growth_rate, nb_layers)

    def _make_layer(self, block, in_planes, growth_rate=24, nb_layers=5):
        layers = []
        for i in range(nb_layers):
            layers.append(block(in_planes + i * growth_rate, growth_rate, 3))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class SimpleAE(nn.Module):
    def __init__(self, nb_layers, in_planes, n_f):
        super(SimpleAE, self).__init__()
        layers = []
        df = in_planes
        for i in range(nb_layers):
            layers.append(nn.BatchNorm2d(df))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(df, n_f, 3, 1, 1))
            df = n_f
        layers.append(nn.BatchNorm2d(df))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(df, 1, 3, 1, 1))
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)


class DeepInSAR(AutoModel):

    def __init__(self, hparams, device, *args, **kwargs):
        super(DeepInSAR, self).__init__()
        self.hparams = hparams
        unit = 5

        self.fe = nn.Sequential(nn.Conv2d(4, hparams.fe_dim * 2, kernel_size=7, stride=1, padding=3), DenseBlock(unit, hparams.fe_dim * 2, 24, BasicBlock))
        self.modules['fe'] = nn.DataParallel(self.fe)

        fe_dim = hparams.fe_dim * 2 + 24 * unit 

        if (not self.hparams.heavy_config):
            self.ae_i = nn.DataParallel(SimpleAE(unit*2, fe_dim, 24))
            self.modules['ae_i'] = (self.ae_i)
            self.ae_r = nn.DataParallel(SimpleAE(unit*2, fe_dim, 24))
            self.modules['ae_r'] = (self.ae_r)
            self.coh  = nn.DataParallel(SimpleAE(unit*2, fe_dim, 24))
            self.modules['coh'] = (self.coh)
        else:
            self.ae_phase = nn.DataParallel(DenseBlock(unit, fe_dim, 24, BasicBlock))
            self.modules['ae_phase'] = (self.ae_phase)

            self.real_out = nn.DataParallel(nn.Conv2d(fe_dim + 24 * unit, 1, 3, 1, 1))
            self.modules['real_out'] = (self.real_out)

            self.imag_out = nn.DataParallel(nn.Conv2d(fe_dim + 24 * unit, 1, 3, 1, 1))
            self.modules['imag_out'] = (self.imag_out)

            self.ae_coh = nn.DataParallel((DenseBlock(unit, fe_dim, 24, BasicBlock)))
            self.modules['ae_coh'] = (self.ae_coh)
            self.coh_out = nn.DataParallel(nn.Conv2d(fe_dim + 24 * unit, 1, 3, 1, 1))
            self.modules['coh_out'] =(self.coh_out)

        self.device = device

        self.to(device)
        self.optimizer = optim.Adam(self.all_parameters(), hparams.lr)
        if (self.hparams.coh_loss == 'MSE'):
            self.cohCre = nn.BCEWithLogitsLoss()

    def all_parameters(self):
        params = []
        for mn in self.modules.keys():
            params += list(self.modules[mn].parameters())
        return params

    def forward(self, x):
        features = self.fe(x)

        if (not self.hparams.heavy_config):
            real = self.ae_r(features)
            imag = self.ae_i(features)
            coh = self.coh(features)
        else:
            real = self.real_out(self.ae_phase(features))
            imag = self.imag_out(self.ae_phase(features))
            coh = self.coh_out(self.ae_coh(features))

        return real, imag, coh

    def fit_step(self, batch):
        (i, batch_data) = batch

        observation = torch.cat([batch_data['ifg_real'], batch_data['ifg_imag'], batch_data['slc1'], batch_data['slc2']], dim=1).to(self.device)

        clean_real = batch_data['clean_real'].to(self.device)
        clean_imag = batch_data['clean_imag'].to(self.device)
        target_coh = batch_data['coh'].to(self.device)


        # run forward 
        real, imag, coh = self.forward(observation)

        loss_r = ((clean_real - observation[:,0,:,:] - real)**2).mean()/2
        loss_i = ((clean_imag - observation[:,1,:,:] - imag)**2).mean()/2
        if (self.hparams.coh_thresh > 0):
            target_coh[target_coh<self.hparams.coh_thresh] = 0
            target_coh[target_coh>=self.hparams.coh_thresh] = 1
        if (self.hparams.coh_loss == 'MSE'):
            loss_coh = ((clean_imag - torch.sigmoid(coh))**2).mean()/2
        else:
            loss_coh = self.cohCre(coh, target_coh)
        
        loss_all = loss_r + loss_i + loss_coh

        self.optimizer.zero_grad()

        loss_all.backward()
        
        self.optimizer.step()

        return {'pbar_desc':'r_loss:{:.3f}, i_loss:{:.3f}, coh_loss:{:.3f}'.format(loss_r.item(), loss_i.item(), loss_coh.item())} 

    @staticmethod
    def add_args(parser):
        parser.add_argument('--lr', default=0.003, type=float)
        parser.add_argument('--grad_acc', default=1, type=float)
        parser.add_argument('--fe_dim', default=24, type=float)
        parser.add_argument('--coh_loss', default='MSE', type=str)
        parser.add_argument('--coh_thresh', default=0, type=float)
        parser.add_argument('--heavy_config', action='store_true')
        return parser

    def all_module_state(self):
        state = {}
        for mn in self.modules.keys():
            state[mn] = self.modules[mn].state_dict()
        return state

    def on_save_checkpoint(self, ckpt_dir):
        module_state_path = '{}/Model.pth'.format(ckpt_dir) 
        torch.save({'modules': self.all_module_state(), 'opt': self.optimizer.state_dict()}, module_state_path)

