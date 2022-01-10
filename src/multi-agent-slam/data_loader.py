# Convert hokuyo_30m from bin to numpy file.
from distributed_sparse_gp.perf import Timer
from numpy import random
import numpy as np
import os
import scipy
import struct

def convert(x_s):
    scaling = 0.005 # 5 mm
    offset = -100.0
    x = x_s * scaling + offset
    return x

def transform_body_to_h30(pos_gt):
    """
    Transform from ground truth to Hokuyo H30 frame.
    :pos_gt: the coordinates, n x 3, in the ground-truth frame. x, y, yaw.
    """
    Rx = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    pos_h30 = Rx @ pos_gt.T
    return pos_h30.T

class GroundTruthLoader:

    def __init__(self, gt_path, cov_path):
        self.gt_path = gt_path
        self.cov_path = cov_path
        self.gt = None
        self.cov = None
        self.interp = None

    def load(self):
        self.gt = np.loadtxt(self.gt_path, delimiter=",")
        self.cov = np.loadtxt(self.cov_path, delimiter=",")
        self.interp = scipy.interpolate.interp1d(self.gt[:, 0], self.gt[:, 1:], kind='nearest', axis=0, fill_value="extrapolate")

    def get_pos(self, time):
        """
        Interpolate and get the ground-truth corresponding to the time.
        :param time: the time to get the ground truth at.
        :return: The position of the robot.
        """
        pose_gt = self.interp(time)
        return transform_body_to_h30(pose_gt[[0, 1, 5]]) # x, y, yaw

    def get_length(self):
        if self.gt is not None:
            return self.gt.shape[0]
        else:
            return 0

class Hokuyo30mLoader:

    def __init__(self, data_path, down_sampling=1.0):
        self.data_path = data_path
        self.num_hits = 1081
        self.down_sampling = down_sampling

        # angles for each range observation
        self.laser_start = -135 * (np.pi / 180.0)
        self.step = 0.25 * (np.pi / 180.0)
        self.angles = np.linspace(self.laser_start, self.laser_start + (self.num_hits - 1) * self.step, self.num_hits)
        self.f_stream = None

    def load(self):
        self.f_stream = open(self.data_path, "rb")

    def get_next(self):
        next_readings = self.get_next_reading()
        while (random.uniform(0.0, 1.0) > self.down_sampling and next_readings is not None):
            next_readings = self.get_next_reading()
        return next_readings

    def get_next_reading(self):
        t_byte = self.f_stream.read(8)
        if t_byte:
            # has timestamp
            utime = struct.unpack('<Q', t_byte)[0]
            # print('Timestamp', utime)
            dist = np.zeros(self.num_hits)
            for i in range(self.num_hits):
                s = struct.unpack('<H', self.f_stream.read(2))[0]
                # s = f_bin.read(2)
                # print(s)
                dist[i] = convert(s)
                # print("Shape of observation: %s" % (dist.shape,))
            return utime, self.angles, dist
        else:
            # reach the end of the line.
            return None

    def close(self):
        self.f_stream.close()

class NCLTSequenceLoader:

    def __init__(self, dataset_path, date, down_sampling):
        self.date = date

        # directory for sequence
        sequence_directory = os.path.join(dataset_path, date)
        self.gt_path = os.path.join(sequence_directory, "groundtruth_{}.csv".format(date))
        self.cov_path = os.path.join(sequence_directory, "cov_{}.csv".format(date))
        self.laser_path = os.path.join(sequence_directory, "{}_hokuyo/{}/hokuyo_30m.bin".format(date, date))
        self.gt_loader = GroundTruthLoader(self.gt_path, self.cov_path)
        self.laser_loader = Hokuyo30mLoader(self.laser_path, down_sampling)

    def get_next(self):
        obs = self.laser_loader.get_next()
        if obs is not None:
            t, angle, dist = obs
            pos = self.gt_loader.get_pos(t)
            return t, dist, angle, pos
        else:
            return None

    def load(self):
        print("Loading groundtruth from: %s; covariance from: %s" % (self.gt_path, self.cov_path))
        self.gt_loader.load()
        self.laser_loader.load()

    def close(self):
        self.gt_loader.close()
        self.laser_loader.close()

    def get_length(self):
        return self.gt_loader.get_length()

def test_loader():
    date = "2012-04-29"
    input_format = "/home/jamesdi1993/datasets/NCLT/{}/{}_hokuyo/{}/hokuyo_30m.bin"
    input_path = input_format.format(date, date, date)
    print("Input path: %s;" % (input_path))
    loader = Hokuyo30mLoader(input_path)
    loader.open()

    t = 10
    for i in range(t):
        obs = loader.get_next()
        print("observation: %s" % obs)
    loader.close()

if __name__ == "__main__":
    test_loader()