import logging

class AutoLogger(object):

    def __init__(self):
        self.initialized = False
        self._pyLogger = None
        self._tbWriter = None

    def initialize(self, log_dir, tensorboard = False):
        self.log_dir = log_dir

        # configure logger
        logFormatter = logging.Formatter('%(asctime)s %(name)-12s %(filename)s:%(lineno)s - %(funcName)-12s  %(levelname)-8s %(message)s')
        self._pyLogger = logging.getLogger()
        fileHandler = logging.FileHandler('{}/stdout.log'.format(self.log_dir))
        fileHandler.setFormatter(logFormatter)
        self._pyLogger.addHandler(fileHandler)
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        self._pyLogger.addHandler(consoleHandler)
        self._pyLogger.setLevel(logging.DEBUG)

        if tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self._tbWriter = SummaryWriter(log_dir=log_dir)

    def tbWriter(self):
        return self._tbWriter

    def pyLogger(self):
        return self._pyLogger


