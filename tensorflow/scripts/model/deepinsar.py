import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from ..data.data_utils import ap2one
import time
import os
MODEL_VERSION = 'DeepInSAR_SIM_DEMO'
def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

class BaseModel(object):
    def __init__(self, config):
        self.config = config

        self.pi = np.pi
        self.build_graph()

        self.saver = tf.train.Saver(max_to_keep=10)
        self.writer = tf.summary.FileWriter(self.config.logs_dir)

        print("[%s]: Model initialized" % self.__class__.__name__)


    def build_graph(self):
        # Define inputs
        pass
    def build_ae(self, net, is_train=True, reuse=False, name="ae"):
        pass

    def build_dense_ae(self, net, is_train=True, reuse=False, name="dense_ae"):
        pass

    def update(self, sess, batch_ifgs, batch_filts, batch_cohs, iter_num):
        pass 

    def inference(self, img, slc1, slc2, sess):
        pass

    def load(self, sess, ckpt_dir):
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt:
            full_path = tf.train.latest_checkpoint(ckpt_dir)
            global_step = int(full_path.split('/')[-1].split('-')[-1])
            self.saver.restore(sess, full_path)
            return True, global_step
        else:
            return False, 0

    def save(self, sess, ckpt_dir, iter_num):
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        print("[%s]: save model" % self.__class__.__name__)
        self.saver.save(sess, os.path.join(ckpt_dir, MODEL_VERSION), global_step=iter_num, write_meta_graph=False)

    def sample(self, sess, img, slc1, slc2, iter_num):
        [h, w] = img.shape
        img = img.reshape([1, 1, h, w, 1])
        slc1 = slc1.reshape([1, 1, h, w, 1])
        slc2 = slc2.reshape([1, 1, h, w, 1])
        recon_images, cohs = self.inference(img, slc1, slc2, sess)
        file_name = "%s/%d_filt.png" % (self.config.sample_dir, iter_num)
        plt.imsave(file_name, np.reshape(recon_images, [h, w]), cmap="jet")
        if cohs is not None:
            file_name = "%s/%d_coh.png" % (self.config.sample_dir, iter_num)
            plt.imsave(file_name, np.reshape(cohs, [h, w]), cmap="gray", vmax=1, vmin=0)
        return 

    def evl(self, img, slc1, slc2):
        [h, w] = img.shape
        img = img.reshape([1, 1, h, w, 1])
        slc1 = slc1.reshape([1, 1, h, w, 1])
        slc2 = slc2.reshape([1, 1, h, w, 1])
        recon_images, coh = self.inference(img, slc1, slc2, None)
        return recon_images, coh 



class Model(BaseModel):
    def build_graph(self):
        # Define inputs
        self.batch_clean_r = tf.placeholder(tf.float32, [None, None, None, None, 1], name="clean_r")
        self.batch_noisy_r = tf.placeholder(tf.float32, [None, None, None, None, 1], name="noisy_r")

        self.batch_clean_i = tf.placeholder(tf.float32, [None, None, None, None, 1], name="clean_i")
        self.batch_noisy_i = tf.placeholder(tf.float32, [None, None, None, None, 1], name="noisy_i")

        self.phase_noisy = tf.atan2(self.batch_noisy_i, self.batch_noisy_r) / np.pi
        self.phase_clean = tf.atan2(self.batch_clean_i, self.batch_clean_r) / np.pi

        self.batch_slc1_amp = tf.placeholder(tf.float32, [None, None, None, None, 1], name="slc1_amp")
        self.batch_slc2_amp = tf.placeholder(tf.float32, [None, None, None, None, 1], name="slc2_amp")


        self.batch_cohs = tf.placeholder(tf.float32, [None, None, None, None, 1], name="cohs")

        self.lr = tf.placeholder(tf.float32)

        tower_grads_fe_ae_r = []
        tower_grads_fe_ae_i = []
        tower_grads_fe_ae_cohs = []
        reuse = False
        ae_r_optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.9)
        ae_i_optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.9)
        ae_coh_optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.9)

        for i in range(self.config.args.num_device):
            with tf.device('/%s:%d' % ("gpu", i)):
                with tf.name_scope('T_%d' % i) as scope:
                    self.forward(slice_index=i, is_train=True, reuse=reuse)
                    reuse = True
                    fe_vars = tf.trainable_variables("feature_extractor")
                    ae_r_vars = tf.trainable_variables("dae_r")
                    ae_i_vars = tf.trainable_variables("dae_i")
                    ae_coh_vars = tf.trainable_variables("ae_coh")

                    # Collect gradients
                    grads_fe_ae_r = ae_r_optimizer.compute_gradients(self.ae_loss_r, var_list=[fe_vars, ae_r_vars])
                    tower_grads_fe_ae_r.append(grads_fe_ae_r)
                    grads_fe_ae_i = ae_i_optimizer.compute_gradients(self.ae_loss_i, var_list=[fe_vars, ae_i_vars])
                    tower_grads_fe_ae_i.append(grads_fe_ae_i)
                    grads_fe_ae_coh = ae_coh_optimizer.compute_gradients(self.ae_loss_coh, var_list=[ae_coh_vars])
                    tower_grads_fe_ae_cohs.append(grads_fe_ae_coh)
        self.forward(0, is_train=False, reuse=True)

        grads_fe_ae_r = average_gradients(tower_grads_fe_ae_r)
        grads_fe_ae_i = average_gradients(tower_grads_fe_ae_i)
        grads_fe_ae_coh = average_gradients(tower_grads_fe_ae_cohs)

        def clip(grads_vars):
            grads, var = zip(*grads_vars)
            grads, global_norm = tf.clip_by_global_norm(grads, 5)
            return zip(grads, var)

        grads_fe_ae_r = clip(grads_fe_ae_r)
        grads_fe_ae_i = clip(grads_fe_ae_i)
        grads_fe_ae_coh = clip(grads_fe_ae_coh)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.ae_r_optimizer = ae_r_optimizer.apply_gradients(grads_fe_ae_r)
            self.ae_i_optimizer = ae_i_optimizer.apply_gradients(grads_fe_ae_i)
            self.ae_coh_optimizer = ae_coh_optimizer.apply_gradients(grads_fe_ae_coh)

        with tf.name_scope('summary'):
            summary_list = []
            summary_list.append(tf.summary.scalar("ae_r", self.ae_loss_r, family="Loss"))
            summary_list.append(tf.summary.scalar("ae_i", self.ae_loss_i, family="Loss"))
            summary_list.append(tf.summary.scalar("ae_coh", self.ae_loss_coh, family="Loss"))

            summary_list.append(tf.summary.image("recon", self.output_phase, max_outputs=2, family="Pair"))
            summary_list.append(tf.summary.image("clean", self.phase_clean[-1], max_outputs=2, family="Pair"))
            summary_list.append(tf.summary.image("noisy", self.phase_noisy[-1], max_outputs=2, family="Pair"))

            self.merged_summary = tf.summary.merge(summary_list)

    def forward(self, slice_index=0, is_train=True, reuse=False):
        # SAE Part
        if is_train is True:
            self.features = self.build_dense_feature_extractor(
                tf.concat(
                    [self.batch_noisy_r[slice_index], self.batch_noisy_i[slice_index], self.batch_slc1_amp[slice_index], self.batch_slc2_amp[slice_index]],
                    axis=-1),
                is_train=True,
                reuse=reuse,
                name="feature_extractor",
                depth=self.config.dfe_depth,
                ef_dim=self.config.dfe_dim,
                g_rate=self.config.g_rate)
            self.ae_r = self.batch_noisy_r[slice_index] - self.build_ae(
                self.features, is_train=True, reuse=reuse, name="dae_r", depth=self.config.fe_depth, ef_dim=self.config.fe_dim)
            self.ae_i = self.batch_noisy_i[slice_index] - self.build_ae(
                self.features, is_train=True, reuse=reuse, name="dae_i", depth=self.config.fe_depth, ef_dim=self.config.fe_dim)
            self.ae_coh = self.build_ae(self.features, is_train=True, reuse=reuse, name="ae_coh",depth=self.config.fe_depth, ef_dim=self.config.fe_dim)
            self.output_phase = tf.atan2(self.ae_i, self.ae_r) / np.pi
            self.ae_loss_r = (1 / 2.0) * tf.reduce_mean(tf.reduce_mean(tf.squared_difference(self.ae_r, self.batch_clean_r[slice_index]), axis=[1, 2, 3]))
            self.ae_loss_i = (1 / 2.0) * tf.reduce_mean(tf.reduce_mean(tf.squared_difference(self.ae_i, self.batch_clean_i[slice_index]), axis=[1, 2, 3]))
            
            if (self.config.coh_loss == 'BCE'):
                self.ae_loss_coh = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.batch_cohs[slice_index], logits=self.ae_coh))
            else:
                # MSE also works 
                self.ae_loss_coh = (1 / 2.0) * tf.reduce_mean(tf.reduce_mean(tf.squared_difference(tf.sigmoid(self.ae_coh) , self.batch_cohs[slice_index]), axis=[1, 2, 3]))

        else:
            self.features_evl = self.build_dense_feature_extractor(
                tf.concat(
                    [self.batch_noisy_r[slice_index], self.batch_noisy_i[slice_index], self.batch_slc1_amp[slice_index], self.batch_slc2_amp[slice_index]],
                    axis=-1),
                is_train=False,
                reuse=True,
                name="feature_extractor",
                depth=self.config.dfe_depth,
                ef_dim=self.config.dfe_dim,
                g_rate=self.config.g_rate)

            self.ae_r_evl = self.batch_noisy_r[slice_index] - self.build_ae(
                self.features_evl, is_train=False, reuse=True, name="dae_r", depth=self.config.fe_depth, ef_dim=self.config.fe_dim)
            self.ae_i_evl = self.batch_noisy_i[slice_index] - self.build_ae(
                self.features_evl, is_train=False, reuse=True, name="dae_i", depth=self.config.fe_depth, ef_dim=self.config.fe_dim)

            self.ae_coh_evl = tf.nn.sigmoid(self.build_ae(self.features_evl, is_train=False, reuse=True, name="ae_coh",depth=self.config.fe_depth, ef_dim=self.config.fe_dim))
            self.output_phase_evl = tf.atan2(self.ae_i_evl, self.ae_r_evl) / np.pi

    def build_ae(self, net, is_train=True, reuse=False, name="dense_ae", depth=10, ef_dim=24):
        d_ef_dim = ef_dim
        # net_list.append(net)
        with tf.variable_scope(name, reuse=reuse):
            for i in range(2, depth):
                net = tf.layers.batch_normalization(net, training=is_train, reuse=reuse, name="bn_" + str(i) + "_0")
                net = tf.nn.relu(net)
                net = tf.layers.conv2d(
                    net, filters=d_ef_dim, kernel_size=3, strides=(1, 1), padding='same', activation=tf.identity, reuse=reuse, name="conv_" + str(i) + "_0")
            net = tf.layers.batch_normalization(net, training=is_train, reuse=reuse, name="bn_o")
            net = tf.nn.relu(net)
            net = tf.layers.conv2d(net, filters=1, kernel_size=3, strides=(1, 1), padding='same', activation=tf.identity, reuse=reuse, name="out")
        return net

    def build_dense_feature_extractor(self, net, is_train=True, reuse=False, name="feature_extractor", depth=6, ef_dim=24, g_rate=3):
        d_ef_dim = ef_dim
        with tf.variable_scope(name, reuse=reuse):
            net = tf.layers.conv2d(net, filters=d_ef_dim * 2, kernel_size=7, strides=(1, 1), padding='same', activation=tf.identity, reuse=reuse, name="input")
            features = net
            for i in range(2, depth):
                net = tf.layers.batch_normalization(features, training=is_train, reuse=reuse, name="bn_" + str(i) + "_0")
                net = tf.nn.relu(net)
                net = tf.layers.conv2d(
                    net,
                    filters=int(d_ef_dim * g_rate),
                    kernel_size=1,
                    strides=(1, 1),
                    padding='same',
                    activation=tf.identity,
                    reuse=reuse,
                    name="botl_" + str(i) + "_0")
                net = tf.layers.batch_normalization(net, training=is_train, reuse=reuse, name="bn_" + str(i) + "_1")
                net = tf.nn.relu(net)
                net = tf.layers.conv2d(
                    net, filters=d_ef_dim, kernel_size=3, strides=(1, 1), padding='same', activation=tf.identity, reuse=reuse, name="conv_" + str(i) + "_0")
                features = tf.concat([features, net], axis=-1)
        return features

    def update(self, sess, batch_ifgs, batch_slc1, batch_slc2, batch_filts, batch_cohs, iter_num):
        _, batch_clean_r, batch_clean_i = ap2one(batch_filts)

        _, batch_noisy_r, batch_noisy_i = ap2one(batch_ifgs)

        batch_clean_r = (batch_clean_r + 1) / 2
        batch_clean_i = (batch_clean_i + 1) / 2
        batch_noisy_r = (batch_noisy_r + 1) / 2
        batch_noisy_i = (batch_noisy_i + 1) / 2

        batch_scl1_amp = np.absolute(batch_slc1)
        batch_scl2_amp = np.absolute(batch_slc2)
        if self.config.coh_thresh > 0:
            batch_cohs[batch_cohs > self.config.coh_thresh] = 1
            batch_cohs[batch_cohs <= self.config.coh_thresh] = 0

        lr = self.config.init_lr
        if iter_num > 1e5:
            lr /= 100.0
        # if iter_num > 1e6:
        #     lr /= 100.0
        [_, _, _, ae_loss_r, ae_loss_i, ae_loss_coh, summary] = sess.run(
            [self.ae_r_optimizer, self.ae_i_optimizer, self.ae_coh_optimizer, self.ae_loss_r, self.ae_loss_i, self.ae_loss_coh, self.merged_summary],
            feed_dict={
                self.batch_noisy_r: batch_noisy_r,
                self.batch_noisy_i: batch_noisy_i,
                self.batch_clean_r: batch_clean_r,
                self.batch_clean_i: batch_clean_i,
                self.batch_slc1_amp: batch_scl1_amp,
                self.batch_slc2_amp: batch_scl2_amp,
                self.batch_cohs: batch_cohs,
                self.lr: lr
            })
        if iter_num % self.config.summary_step == 0:
            self.writer.add_summary(summary, global_step=iter_num)
            print("[%s]: write summary" % self.__class__.__name__)
        log_message = "ae_loss_r=%.3f, ae_loss_i =%.3f, ae_loss_coh =%.3f" % (ae_loss_r, ae_loss_i, ae_loss_coh)
        return log_message

    def inference(self, ifgs, slc1, slc2, sess=None):
        print("inference")
        _, batch_noisy_r, batch_noisy_i = ap2one(np.copy(ifgs))
        batch_noisy_r = (batch_noisy_r + 1) / 2
        batch_noisy_i = (batch_noisy_i + 1) / 2
        batch_scl1_amp = np.absolute(slc1)
        batch_scl2_amp = np.absolute(slc2)

        if sess is None:
            sess = tf.get_default_session()
        t = time.time()
        [recon_r, recon_i, cohs] = sess.run(
            [self.ae_r_evl, self.ae_i_evl, self.ae_coh_evl],
            feed_dict={
                self.batch_noisy_r: batch_noisy_r,
                self.batch_noisy_i: batch_noisy_i,
                self.batch_slc1_amp: batch_scl1_amp,
                self.batch_slc2_amp: batch_scl2_amp,
            })
        recon_r = recon_r * 2 - 1
        recon_i = recon_i * 2 - 1
        elapsed = time.time() - t
        print("Processing time:%.8f" % (elapsed))
        print("end")
        recon_ifg = recon_r + 1j * recon_i
        recon_images = np.angle(recon_ifg)
        return recon_images, cohs

