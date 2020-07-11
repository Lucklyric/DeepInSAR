import glob
import multiprocessing
import re
import sys
import threading
import time

import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize

from .data_utils import readFloat, readFloatComplex, readFloatComplexRandomPathces, readFloatRandomPathces, readShortFloatComplexRandomPathces, InSARDB, parse_json_db


class QThread(threading.Thread):
    def __init__(self, name, target):
        threading.Thread.__init__(self, name=name, target=target)
        self.running = True
        self.run_lock = threading.Lock()

    def stop_thread(self):
        self.run_lock.acquire()
        self.running = False
        self.run_lock.release()

    def running_state(self):
        self.run_lock.acquire()
        state = self.running
        self.run_lock.release()
        return state


class BytesChannel(object):
    def __init__(self, maxsize):
        self.buffer_ifg = multiprocessing.RawArray(np.ctypeslib.ctypes.c_ubyte, maxsize * 8 + 1)
        self.buffer_filt = multiprocessing.RawArray(np.ctypeslib.ctypes.c_ubyte, maxsize * 8 + 1)
        self.buffer_slc1 = multiprocessing.RawArray(np.ctypeslib.ctypes.c_ubyte, maxsize * 8 + 1)
        self.buffer_slc2 = multiprocessing.RawArray(np.ctypeslib.ctypes.c_ubyte, maxsize * 8 + 1)
        self.buffer_coh = multiprocessing.RawArray(np.ctypeslib.ctypes.c_ubyte, maxsize * 4 + 1)
        self.buffer_len = multiprocessing.Value("i")
        self.checking = multiprocessing.Value("i")
        self.empty = multiprocessing.Semaphore(1)
        self.full = multiprocessing.Semaphore(0)
        self.checking.value = 0

    def send(self, ifgs, slc1, slc2, filts, cohs):
        self.empty.acquire()
        nitems = len(ifgs)
        self.buffer_len.value = nitems
        self.buffer_filt[:nitems] = filts
        self.buffer_ifg[:nitems] = ifgs
        self.buffer_slc1[:nitems] = slc1
        self.buffer_slc2[:nitems] = slc2
        self.buffer_coh[:int(nitems / 2)] = cohs
        self.full.release()

    def recv(self):
        self.full.acquire()
        ifgs = self.buffer_ifg[:self.buffer_len.value]
        slc1 = self.buffer_slc1[:self.buffer_len.value]
        slc2 = self.buffer_slc2[:self.buffer_len.value]
        filts = self.buffer_filt[:self.buffer_len.value]
        cohs = self.buffer_coh[:int(self.buffer_len.value / 2)]
        self.empty.release()
        return ifgs, slc1, slc2, filts, cohs


class DataReaderPatchWiseMP(object):
    def __init__(self,
            insar_dbs,
            batch_size=64,
            patch_size=64,
            num_sample_db_per_run=1,
            num_sample_img_per_db=10,
            num_sample_patch_per_img=30,
            num_process=2,
            min_cap_of_patches=2000,
            max_cap_of_patches=8000,
            verbose=True):
        self.ifgs_q = []
        self.filts_q = []
        self.cohs_q = []
        self.slc1_q = []
        self.slc2_q = []
        self.insar_dbs = insar_dbs
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.num_sample_db_per_run = num_sample_db_per_run
        self.num_img_per_db = num_sample_img_per_db
        self.num_sample_patch_per_img = num_sample_patch_per_img
        self.lock = threading.Lock()
        self.q_process = []
        self.min_cap_of_patches = min_cap_of_patches
        self.max_cap_of_patches = max_cap_of_patches
        self.shuffle_period = -1
        self.verbose = verbose
        self.fetch_thread = QThread("fetch", target=self.fetching_loop)
        self.total_sample = self.num_sample_db_per_run * self.num_img_per_db * self.num_sample_patch_per_img
        self.bytes_channel = BytesChannel(self.total_sample * patch_size * patch_size)
        for _ in range(num_process):
            self.q_process.append(multiprocessing.Process(target=self.run_loop, args=([self.bytes_channel])))

    def append_to_q(self, ifgs, slc1, slc2, filts, cohs):
        self.lock.acquire()
        self.ifgs_q.extend(ifgs)
        self.slc1_q.extend(slc1)
        self.slc2_q.extend(slc2)
        self.filts_q.extend(filts)
        self.cohs_q.extend(cohs)
        self.lock.release()
        if self.verbose:
            print("[%s]: Append data from_thread %s, current Q len=%d" % (self.__class__.__name__, threading.current_thread().getName(), len(self.cohs_q)))
        # print(threading.current_thread().running_state())
        return

    def pop_from_q(self):
        while True:
            if len(self.cohs_q) > self.min_cap_of_patches:
                if self.verbose:
                    print("[%s]: %d left in Q" % (self.__class__.__name__, len(self.cohs_q)))
                break
            time.sleep(1 / 120.0)
        self.lock.acquire()
        current_len = len(self.ifgs_q)
        idxes = (np.random.randint(0,current_len,self.batch_size)).tolist()

        batch_ifgs = [self.ifgs_q[x] for x in idxes]
        batch_filts = [self.filts_q[x] for x in idxes]
        batch_cohs = [self.cohs_q[x] for x in idxes]
        batch_slc1 = [self.slc1_q[x] for x in idxes]
        batch_slc2 = [self.slc2_q[x] for x in idxes]
        print(current_len)
        if current_len >= self.max_cap_of_patches:
            del self.ifgs_q[:int(self.max_cap_of_patches/2)]
            del self.filts_q[:int(self.max_cap_of_patches/2)]
            del self.slc1_q[:int(self.max_cap_of_patches/2)]
            del self.slc2_q[:int(self.max_cap_of_patches/2)]
            del self.cohs_q[:int(self.max_cap_of_patches/2)]
        self.lock.release()
        return batch_ifgs, batch_slc1, batch_slc2, batch_filts, batch_cohs

    def fetching_loop(self):
        t = threading.current_thread()
        while t.running_state():
            if len(self.cohs_q) < self.max_cap_of_patches:
                self.bytes_channel.checking.value = 0
                ifgs, slc1, slc2, filts, cohs = self.bytes_channel.recv()
                ifgs = np.reshape(np.frombuffer(bytearray(ifgs), dtype=">c8").astype(complex), [self.total_sample, self.patch_size, self.patch_size, 1])
                filts = np.reshape(np.frombuffer(bytearray(filts), dtype=">c8").astype(complex), [self.total_sample, self.patch_size, self.patch_size, 1])
                slc1 = np.reshape(np.frombuffer(bytearray(slc1), dtype=">c8").astype(complex), [self.total_sample, self.patch_size, self.patch_size, 1])
                slc2 = np.reshape(np.frombuffer(bytearray(slc2), dtype=">c8").astype(complex), [self.total_sample, self.patch_size, self.patch_size, 1])
                cohs = np.reshape(np.frombuffer(bytearray(cohs), dtype=">f4").astype(float), [self.total_sample, self.patch_size, self.patch_size, 1])
                self.append_to_q(ifgs, slc1, slc2, filts, cohs)
            else:
                self.bytes_channel.checking.value = 1
            # time.sleep(1 / 60.0)

    def run_loop(self, ch):
        while True:
            if ch.checking.value == 0:
                try:
                    ifgs, slc1, slc2, filts, cohs = self.prepare_data()
                    ch.send(
                            bytearray(ifgs.flatten().astype('>c8')), bytearray(slc1.flatten().astype('>c8')), bytearray(slc2.flatten().astype('>c8')),
                            bytearray(filts.flatten().astype('>c8')), bytearray(cohs.flatten().astype('>f4')))
                except ValueError:
                    print("value Error")
            else:
                time.sleep(1)  # save cpu usage
            # time.sleep(1 / 60.0)

    def prepare_data(self):
        db_idxes = np.random.randint(0, len(self.insar_dbs), size=[self.num_sample_db_per_run])
        ifgs_patches = []
        slc1_patches = []
        slc2_patches = []
        filts_patches = []
        cohs_patches = []
        for db_idx in db_idxes:
            db = self.insar_dbs[db_idx]
            file_idxes = np.random.randint(0, len(db.noisy_files), size=[self.num_img_per_db])
            for f_idx in file_idxes:
                # Read all imgs
                # print(threading.current_thread().getName(),f_idx)
                ifgs, rs, cs, h = readFloatComplexRandomPathces(
                        db.noisy_files[f_idx], width=db.width, patch_size=self.patch_size, num_sample=self.num_sample_patch_per_img)
                slc1_path, slc2_path = db.get_rslcs_paths(db.noisy_files[f_idx])

                slc1, rs, cs, h = readFloatComplexRandomPathces(
                        slc1_path, width=db.width, patch_size=self.patch_size, num_sample=self.num_sample_patch_per_img,rows=rs, cols=cs, height=h)
                slc2, rs, cs, h = readFloatComplexRandomPathces(
                        slc2_path, width=db.width, patch_size=self.patch_size, num_sample=self.num_sample_patch_per_img,rows=rs, cols=cs, height=h)
                filts, rs, cs, h = readFloatComplexRandomPathces(
                        db.filt_files[f_idx], width=db.width, patch_size=self.patch_size, num_sample=self.num_sample_patch_per_img, rows=rs, cols=cs, height=h)
                cohs, rs, cs, h = readFloatRandomPathces(
                        db.coh_files[f_idx], width=db.width, patch_size=self.patch_size, num_sample=self.num_sample_patch_per_img, rows=rs, cols=cs, height=h)
                ifgs_patches.extend(ifgs)
                slc1_patches.extend(slc1)
                slc2_patches.extend(slc2)
                filts_patches.extend(filts)
                cohs_patches.extend(cohs)
        return np.asarray(ifgs_patches), np.asarray(slc1_patches), np.asarray(slc2_patches), np.asarray(filts_patches), np.asarray(cohs_patches)

    def next_batch(self):
        batch = self.pop_from_q()
        ifgs = np.reshape(batch[0], [-1, self.patch_size, self.patch_size, 1])
        slc1 = np.reshape(batch[1], [-1, self.patch_size, self.patch_size, 1])
        slc2 = np.reshape(batch[2], [-1, self.patch_size, self.patch_size, 1])
        filts = np.reshape(batch[3], [-1, self.patch_size, self.patch_size, 1])
        cohs = np.reshape(batch[4], [-1, self.patch_size, self.patch_size, 1])
        return ifgs, slc1, slc2, filts, cohs

    def start_feeding_q(self):
        for i in range(len(self.q_process)):
            np.random.seed(int(time.time()) + i)
            self.q_process[i].start()
        self.fetch_thread.start()

    def stop_feeding_q(self):
        for process in self.q_process:
            process.terminate()
        self.fetch_thread.stop_thread()


if __name__ == '__main__':
    dbs = parse_json_db("./sim_db_config.json")
    test_patch_size = 128
    test_reader = DataReaderPatchWiseMP(
            insar_dbs=dbs,
            batch_size=32,
            patch_size=test_patch_size,
            num_sample_db_per_run=1,
            num_sample_img_per_db=1,
            num_sample_patch_per_img=1,
            num_process=10,
            min_cap_of_patches=100,
            max_cap_of_patches=600)
    test_reader.start_feeding_q()

    [a, slc1, slc2, b, c] = test_reader.next_batch()
    plt.figure()
    plt.imshow(np.reshape(np.angle(a[0]), [test_patch_size, test_patch_size]), cmap="jet")
    plt.figure()
    plt.imshow(np.reshape(np.angle(b[0]), [test_patch_size, test_patch_size]), cmap="jet")
    plt.figure()
    plt.imshow(np.reshape(np.abs((slc1[0]))**0.3, [test_patch_size, test_patch_size]), cmap="gray")
    plt.figure()
    plt.imshow(np.reshape(np.abs((slc2[0]))**0.3, [test_patch_size, test_patch_size]), cmap="gray")
    plt.figure()
    plt.imshow(np.reshape(c[0], [test_patch_size, test_patch_size]), cmap="gray")
    plt.show()
    pass
