from coolname import generate_slug
from datetime import datetime
from tqdm import tqdm
import numpy as np
import os
import typing

import torch

from ..logger import AutoLogger
from ..utils import makedirs

class AutoRunner(object):

    def __init__(self,
            max_num_epochs,
            val_per_iters=0,
            val_per_epoch=0,
            eval_per_iters=0,
            eval_per_epoch=0,
            save_per_iters=0,
            save_per_epoch=0,
            processing_root='./processing',
            ckpt_dir='./ckpts',
            log_dir='./logs',
            eval_dir='./eval',
            example_dir='./examples',
            no_cooldir=False,
            exp_name='exp',
            runner_ckpt_name='runner_state',
            autoLogger=None,
            pretrained_root='',
            train=True,
            new_root=False,
            tensorboard=False,
            *args,
            **kwargs):

        self.model = None
        self.max_num_epochs = max_num_epochs
        self.val_per_iters = val_per_iters
        self.val_per_epoch = val_per_epoch
        self.eval_per_iters = eval_per_iters
        self.eval_per_epoch = eval_per_epoch
        self.save_per_iters = save_per_iters
        self.save_per_epoch = save_per_epoch
        self.runner_ckpt_name = runner_ckpt_name
        self.pretrained_root = pretrained_root
        self.train = train
        self.new_root = new_root
        self.pretrained_ckpt_dir = None 

        create_new_processing_root = False

        # create new processing if no pretrained_root is given 
        if (self.pretrained_root == ''):
            create_new_processing_root = True
        else:
            # assign pretrained checkpoint dir
            self.pretrained_ckpt_dir = '{}/{}'.format(self.pretrained_root, ckpt_dir)

            if (self.train):
                if (not self.new_root):
                    self.processing_root = pretrained_root
                    self.log_dir = '{}/{}'.format(self.processing_root, log_dir)
                    self.example_dir = '{}/{}'.format(self.processing_root, example_dir)
                    self.ckpt_dir = '{}/{}'.format(self.processing_root, ckpt_dir)
                else:        
                    create_new_processing_root = True
            else:
                create_new_processing_root = True
        
        if (create_new_processing_root):
            if (self.train):
                processing_root += '/train/'
            else:
                processing_root += '/test/'

            # configure all dirs
            datestr = datetime.now().strftime("%Y.%m.%d_%H.%M")
            if (no_cooldir):
                tag_name = '{}_{}'.format(exp_name, datestr)
            else:
                tag_name = '{}_{}_{}'.format(exp_name, generate_slug(2), datestr)

            self.processing_root = '{}/{}/'.format(processing_root, tag_name)
            self.log_dir = '{}/{}'.format(self.processing_root, log_dir)
            self.example_dir = '{}/{}'.format(self.processing_root, example_dir)
            self.ckpt_dir = '{}/{}'.format(self.processing_root, ckpt_dir)
            self.eval_dir = '{}/{}'.format(self.processing_root, eval_dir)
            makedirs(self.log_dir)
            makedirs(self.example_dir)
            makedirs(self.ckpt_dir)
            makedirs(self.eval_dir)
        

        if (autoLogger == None):
            self.logger = AutoLogger()
            self.logger.initialize(self.log_dir, tensorboard)
        else:
            self.logger = autoLogger

        self.logger.pyLogger().info('Runner initialized')
        self.logger.pyLogger().info('Status:is_train={}'.format(self.train))
        if (self.pretrained_root != ''):
            self.logger.pyLogger().info('Load pretrained from:{}'.format(self.pretrained_ckpt_dir))
        if (create_new_processing_root):
            self.logger.pyLogger().info('Training with new processing root:{}'.format(self.processing_root))
        else:
            self.logger.pyLogger().info('Continue training at:{}'.format(self.processing_root))

        self.logger.pyLogger().info('Runner max_num_epochs:{}'.format(self.max_num_epochs))



    def fit(self, model, train_dataloader, val_dataloader=None):
        # check if load pretrained model
        self.bind_model(model)
        self.check_pretrain()
        self.model.train()


        for epoch in range(self.model.current_epoch, self.max_num_epochs):
            self.model.current_epoch = epoch
            self.model.on_epoch_begin()
            start_batch_idx = self.model.global_step % len(train_dataloader)
            with tqdm(total=len(train_dataloader), initial=start_batch_idx) as pbar:
                for batch_data in enumerate(train_dataloader):
                    self.model.global_step += 1
                    self.model.on_batch_begin()
                    results = self.model.fit_step(batch_data)
                    if ('pbar_desc' in results):
                        pbar.set_description(results['pbar_desc'])
                    if (self.save_per_iters > 0 and self.model.global_step % self.save_per_iters == 0):
                        self.save_checkpoint(self.ckpt_dir)
                    pbar.update()
                    self.model.on_batch_end()
                    if (self.val_per_iters > 0 and self.model.global_step % self.val_per_iters == 0):
                        self.val(val_dataloader=val_dataloader)
                start_batch_idx = 0
            if (self.save_per_epoch > 0 and self.model.current_epoch + 1 % self.save_per_epoch == 0):
                    self.save_checkpoint(self.ckpt_dir)
            self.model.on_epoch_end()

    def eval(self, model=None, eval_dataloader=None):
        # if mode is none then it is called from fit process
        if (model):
            self.bind_model(model)
            self.check_pretrain()

        if (hasattr(self.model, 'on_eval')):
            self.logger.pyLogger().info('Start evalution')
            self.model.eval()
            results = model.on_eval(eval_dataloader)
            self.model.train()
            return
        pass

    def val(self, model=None, val_dataloader=None):
        # if mode is none then it is called from fit process
        if (model):
            self.bind_model(model)
            self.check_pretrain()

        if (hasattr(self.model, 'on_val')):
            self.logger.pyLogger().info('Start val')
            self.model.eval()
            results = self.model.on_val(val_dataloader)
            self.model.train()
            return results
        pass

    def check_pretrain(self):
        if (self.pretrained_root != ''):
            # load state
            self.load_checkpoint(self.pretrained_ckpt_dir)

    def save_checkpoint(self, ckpt_dir):
        state_dict = {'runner': {'current_epoch': self.model.current_epoch, 'current_iters': self.model.global_step}}
        path = '{}/{}.pth'.format(ckpt_dir, self.runner_ckpt_name)
        torch.save(state_dict,path)
        self.logger.pyLogger().info('Save runner state to %s', os.path.abspath(path))
        self.model.on_save_checkpoint(ckpt_dir)

    def load_checkpoint(self, ckpt_dir):
        self.logger.pyLogger().info('Load runner state from %s', '{}/{}.pth'.format(ckpt_dir, self.runner_ckpt_name))
        runner_state = torch.load('{}/{}.pth'.format(ckpt_dir, self.runner_ckpt_name))
        print(runner_state)
        self.model.current_epoch = runner_state['runner']['current_epoch']
        self.model.global_step = runner_state['runner']['current_current_iters']
        self.model.on_load_checkpoint(ckpt_dir)
    
    def bind_model(self, model):
        self.model = model
        model.set_runner(self)
        model.set_logger(self.logger)

    @staticmethod
    def add_args(parser):
        parser.add_argument('--max_num_epochs', type=int, default=4000, help='max number of epochs')
        parser.add_argument('--save_per_iters', type=int, default=50, help='max number of epochs')
        parser.add_argument('--val_per_iters', type=int, default=50, help='max number of epochs')
        parser.add_argument('--processing_root', type=str, default='./processing', help='str')
        parser.add_argument('--pretrained_root', type=str, default='', help='the dir of a pretrained processing state')
        parser.add_argument('--new_root', action='store_true', help='whether continue the training with the given pretrained_root')
        parser.add_argument('--train', action='store_true', help='whether runner performs training task')
        parser.add_argument('--ckpt_dir', type=str, default='./ckpts', help='str')
        parser.add_argument('--log_dir', type=str, default='./logs', help='str')
        parser.add_argument('--example_dir', type=str, default='./examples', help='str')
        parser.add_argument('--eval_dir', type=str, default='./eval', help='str')
        parser.add_argument('--exp_name', type=str, default='exp', help='str')
        parser.add_argument('--no_cooldir', action='store_true', help='no cool dir name')
        parser.add_argument('--runner_ckpt_name', type=str, default='runner_state', help='str')
        parser.add_argument('--tensorboard', action='store_true', help='enbale tensorboard in autologger')
        return parser

