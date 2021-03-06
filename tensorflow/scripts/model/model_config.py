class ConfigSmall(object):
    def __init__(self, name, args=None):
        self.batch_size = 6 
        self.patch_size = 128
        self.init_lr = 0.001
        self.iteration = 1.6e5
        self.dfe_depth = 6
        self.dfe_dim = 24
        self.g_rate = 3
        self.fe_dim = 24
        self.fe_depth = 6
        self.ckpt_dir = "./record/ckpts/" + name + "/"
        self.sample_dir = "./record/samples/" + name + '/'
        self.logs_dir = "./record/logs/" + name + '/'
        self.evl_dir = "./record/evls/" + name + '/'
        self.summary_step = 1000 
        self.ckpt_step = 1000 
        self.coh_loss = 'MSE'
        self.coh_thresh = 0
        self.ae_dense = False
        self.ae_phase_share = False
        self.args = args

class Config2(object):
    def __init__(self, name, args=None):
        self.batch_size = 32 
        self.patch_size = 128
        self.init_lr = 0.001
        self.iteration = 1.6e5
        self.dfe_depth = 6
        self.dfe_dim = 24
        self.g_rate = 4
        self.fe_dim = 24
        self.fe_depth = 6
        self.ckpt_dir = "./record/ckpts/" + name + "/"
        self.sample_dir = "./record/samples/" + name + '/'
        self.logs_dir = "./record/logs/" + name + '/'
        self.evl_dir = "./record/evls/" + name + '/'
        self.summary_step = 1000 
        self.ckpt_step = 1000 
        self.coh_loss = 'BCE'
        self.coh_thresh = 0.5
        self.ae_dense = False
        self.ae_phase_share = False
        self.args = args

class Config3(object):
    def __init__(self, name, args=None):
        self.batch_size = 64 
        self.patch_size = 128
        self.init_lr = 0.001
        self.iteration = 1.6e5
        self.dfe_depth = 6
        self.dfe_dim = 24
        self.g_rate = 4
        self.fe_dim = 24
        self.fe_depth = 6
        self.ckpt_dir = "./record/ckpts/" + name + "/"
        self.sample_dir = "./record/samples/" + name + '/'
        self.logs_dir = "./record/logs/" + name + '/'
        self.evl_dir = "./record/evls/" + name + '/'
        self.summary_step = 1000 
        self.ckpt_step = 1000 
        self.coh_loss = 'MSE'
        self.coh_thresh = 0
        self.ae_dense = True
        self.ae_phase_share = True
        self.args = args
