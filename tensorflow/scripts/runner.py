import argparse
import importlib
import os
import sys
import os
import shutil
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from .data.dataloader import DataReaderPatchWiseMP
from .data.data_utils import ap2one, parse_json_db, normalize_slc_by_tanhmz, readFloat, readFloatComplex, readShortComplex
from .model.model_config import Config
from .model.deepinsar import Model

# Parsing all arguement and loding corresponding model config db files
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="model_deepInSAR_sim_demo")
parser.add_argument("--config", default="config_B32P128")
parser.add_argument("--runidx", default="r1")
parser.add_argument("--traindb", default="data/sim_db_config.json")
parser.add_argument("--num_device", type=int, default=1)
parser.add_argument("--clear_pre", action="store_true", default=False)
args = parser.parse_args()
print(args)

CONFIG_VERSION = args.config
CONFIG_VERSION = "".join(CONFIG_VERSION.split("_")[1:])

MODEL_VERSION = args.model
MODEL_VERSION = "".join(MODEL_VERSION.split("_")[1:])

TRAINING_VERSION = "ptsl3vGMD"
POSFIX = args.runidx
DIR_PREFIX = "t" + TRAINING_VERSION + "_m" + MODEL_VERSION + "_c" + CONFIG_VERSION + "_" + POSFIX
CONFIG = Config(DIR_PREFIX, args)


def check_dir(path, clear_pre=False):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        print (clear_pre)
        if clear_pre is True:
            shutil.rmtree(path)
            os.makedirs(path)

def main():
    check_dir(CONFIG.ckpt_dir, args.clear_pre)
    check_dir(CONFIG.logs_dir, args.clear_pre)
    check_dir(CONFIG.sample_dir, args.clear_pre)
    check_dir(CONFIG.evl_dir, args.clear_pre)

    train_dbs = parse_json_db(args.traindb)
    data_reader = DataReaderPatchWiseMP(
        insar_dbs=train_dbs,
        batch_size=CONFIG.batch_size,
        patch_size=CONFIG.patch_size,
        num_sample_db_per_run=1,
        num_sample_img_per_db=1,
        num_sample_patch_per_img=1,
        num_process=10,
        min_cap_of_patches=100,
        max_cap_of_patches=8000,
        verbose=False)

    # Start DB
    data_reader.start_feeding_q()

    # Create Model
    model = Model(CONFIG)
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)

    # Start the session
    with tf.Session(config=config) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        _, iter_num = model.load(sess, CONFIG.ckpt_dir)

        while iter_num < CONFIG.iteration:
            tower_batch_noisy = []
            tower_batch_slc1 = []
            tower_batch_slc2 = []
            tower_batch_filts = []
            tower_batch_cohs = []
            for n_d in range(CONFIG.args.num_device):
                batch_noisy, batch_slc1, batch_slc2, batch_filts, batch_cohs = data_reader.next_batch()
                tower_batch_noisy.append(batch_noisy)
                tower_batch_slc1.append(batch_slc1)
                tower_batch_slc2.append(batch_slc2)
                tower_batch_filts.append(batch_filts)
                tower_batch_cohs.append(batch_cohs)

            log_message = model.update(sess, np.asarray(tower_batch_noisy), np.asarray(tower_batch_slc1), np.asarray(tower_batch_slc2),
                                       np.asarray(tower_batch_filts), np.asarray(tower_batch_cohs), iter_num)

            print("Step:[%d] %s" % (iter_num, log_message))
            if iter_num % CONFIG.ckpt_step == 0:
                model.save(sess, CONFIG.ckpt_dir, iter_num)

            iter_num += 1
    os._exit(1)


if __name__ == '__main__':
    try:
        main()
    except (KeyboardInterrupt, SystemExit):
        print("Exiting...")
        os._exit(1)
