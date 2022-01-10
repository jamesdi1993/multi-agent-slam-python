"""
Class for sequencing NCLT dataset files.
"""
from data_loader import Hokuyo30mLoader, GroundTruthLoader
import os
import os.path as path
import numpy as np
import random
import yaml

def load_params(param_path):
    with open(param_path, 'r') as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return params

class NCLTSequencer():
    """
    Class for Sequencing a NCLT Lidar sequence into multiple pieces.
    """
    def __init__(self, data_dir, date, max_step=100000, data_format="csv", down_sampling=1.0):
        self.data_dir = data_dir
        self.date = date
        self.gt_path = path.join(data_dir, "{0}/groundtruth_{0}.csv".format(date))
        self.cov_path = path.join(data_dir, "{0}/cov_{0}.csv".format(date))
        self.lidar_path = path.join(data_dir, "{0}/{0}_hokuyo/{0}/hokuyo_30m.bin".format(date))

        self.gt_loader = GroundTruthLoader(self.gt_path, self.cov_path)
        self.lidar_loader = Hokuyo30mLoader(self.lidar_path)
        self.output_dir = path.join(data_dir, "sequences")
        self.max_step = max_step

        assert data_format in ["csv", "npz"], "data_format not supported: %s" % (data_format)
        self.data_format = data_format
        self.down_sampling = down_sampling
        print("Data format: %s" % self.data_format)
        print("The data sampling rate is: %s" % self.down_sampling)
        print("Sequenced output directory: %s" % self.output_dir)

    def load(self):
        self.gt_loader.load()
        self.lidar_loader.load()

    def extract(self, start, end, sequence_num):
        # output format:
        # laser_t, dist_0, dist_1, ... , dist_1080, x, y, yaw

        output_path = path.join(self.output_dir, "{0}_seq_{1}.{2}".format(self.date, sequence_num, self.data_format))
        t, _, dist = self.lidar_loader.get_next()

        # forward to the next timestamp
        while(t < start):
            t, _, dist = self.lidar_loader.get_next()

        pos = self.gt_loader.get_pos(t).flatten()
        row = np.concatenate(([t], dist.flatten(), pos), axis=0)
        print("Example output: %s" % (row))
        output = [row]
        step = 1

        while (t < end and step < self.max_step):
            next_reading = self.lidar_loader.get_next()
            if next_reading is None:
                break

            if (random.uniform(0.0, 1.0) < self.down_sampling):
                t, _, dist = next_reading
                pos = self.gt_loader.get_pos(t).flatten()
                row = np.concatenate(([t], dist.flatten(), pos), axis=0)
                output.append(row)
                step += 1
                if step % 1000 == 0:
                    print(step)

        if not path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        output_numpy = np.array(output)
        print("number of steps: %s" % (len(output)))
        print("output shape: %s" % (output_numpy.shape,))
        if self.data_format == "csv":
            np.savetxt(output_path, output_numpy, delimiter=",")
        elif self.data_format == "npz":
            np.savez(output_path, output_numpy)
        print("saved seq %d at: %s; Output shape: %s; last timestamp: %s" % (sequence_num, output_path, output_numpy.shape, t))

    def close(self):
        # self.gt_loader.close()
        self.lidar_loader.close()

class NCLTSequenceLoader:

    def __init__(self, data_dir, date, seq_num, down_sampling=1.0, data_format="csv"):
        self.data_dir = os.path.join(data_dir, "sequences")
        self.date = date
        self.data_format = data_format
        assert data_format in ["csv", "npz"], "data_format not supported: %s" % (data_format)

        self.data_path = os.path.join(self.data_dir, "{0}_seq_{1}.{2}".format(date, seq_num, data_format))
        self.down_sampling = down_sampling
        self.dist_numpy = None
        self.sequence_length = 0
        self.pointer = 0
        self.num_hits = 1081
        self.laser_start = -135 * (np.pi / 180.0)
        self.step = 0.25 * (np.pi / 180.0)
        self.angles = np.linspace(self.laser_start, self.laser_start + (self.num_hits - 1) * self.step, self.num_hits)
        
    def load(self):
        if self.data_format == "csv":
            self.dist_numpy = np.loadtxt(self.data_path, delimiter=',')
        elif self.data_format == "npz":
            self.dist_numpy = np.load(self.data_path)['arr_0']
        self.sequence_length = self.dist_numpy.shape[0]
        print("Sequence length: %s" % self.sequence_length)

    def get_next(self):
        next_readings = self.get_next_reading()
        while (random.uniform(0.0, 1.0) > self.down_sampling and next_readings is not None):
            next_readings = self.get_next_reading()
        return next_readings

    def get_next_reading(self):
        if (self.pointer < self.sequence_length):
            obs = self.dist_numpy[self.pointer, :]
            t = obs[0]
            dist = obs[1: 1+ self.num_hits]
            pos = obs[1 + self.num_hits:]
            angles = self.angles
            self.pointer += 1
            return t, dist, angles, pos
        return None

    def has_next(self):
        return self.pointer < self.sequence_length

    def get_length(self):
        return self.sequence_length

def test_csv_loader():

    data_path = "/home/jamesdi1993/datasets/NCLT"
    date = "2012-08-04"

    loader = NCLTSequenceLoader(data_path, date, 1, down_sampling=0.1)
    loader.load()

    while loader.has_next():
        obs = loader.get_next()
        if obs is not None:
            t, dist, angle, pos = obs
            print("The next time is: %s; pos: %s; angle shape: %s; dist shape: %s" % (t, pos, angle.shape, dist.shape))

def extract_sequence(file_path, seq_num, start, data_format, down_sampling=1.0):
    """
    :param file_path: yaml file
    :return:
    """
    data_dir = "/home/jamesdi1993/datasets/NCLT"
    params = load_params(file_path)
    n_agents = params["n_agents"]
    for i in range(start, n_agents):
        agent_param = params.get("agent_{}".format(i))
        if agent_param is not None:
            print("Parsing param for agent: %s; param: %s" % (i, agent_param))
            date = agent_param["date"]
            time_range = agent_param["range"]
            start_time = time_range[0]
            end_time = time_range[1]
            max_step = 400000
            sequencer = NCLTSequencer(data_dir, date, max_step, data_format, down_sampling)
            sequencer.load()
            sequencer.extract(start_time, end_time, seq_num)
            sequencer.close()


if __name__ == "__main__":
    seq_num = 6
    down_sampling = 0.1
    file_path = "/home/jamesdi1993/datasets/NCLT/sequences/seq_{}.yaml".format(seq_num)
    data_format = "npz"
    extract_sequence(file_path, seq_num, 0, data_format, down_sampling)
    # test_csv_loader()
