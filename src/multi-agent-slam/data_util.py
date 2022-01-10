from os import path
import os
import numpy as np
import yaml

def load_params(param_path):
    with open(param_path, 'r') as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return params

class CarmenLoader():

    def __init__(self, file_path):
        self.file_path = file_path
        self.odom = []
        self.front_dist, self.rear_dist = [], []
        self.front_dist_numpy, self.rear_dist_numpy, self.odom_numpy = None, None, None
        self.front_angles, self.rear_angles = [], []
        self.front_angles_numpy, self.rear_angles_numpy = None, None

    def load(self):
        with open(self.file_path) as f:
            line = f.readline()
            while line:
                fields = line.split(" ")
                message_name, data = fields[0], fields[1:]
                if message_name == "FLASER":
                    num_readings = int(data[0])
                    angles = [-np.pi / 2 + i * np.pi / num_readings for i in range(num_readings)]
                    readings = data[1: (1 + num_readings)]

                    assert len(readings) == num_readings, \
                        "num of readings do not match! actual: %s, expected: %s" % (len(readings), num_readings)
                    position = data[(1 + num_readings): (1 + num_readings + 6)]
                    t = data[-1]
                    self.front_dist.append([t] + position + readings)
                    self.front_angles.append(angles)
                elif message_name == "RLASER":
                    num_readings = int(data[0])

                    angles = [-i * np.pi / num_readings for i in range(num_readings)]
                    readings = data[1: (1 + num_readings)]

                    assert len(readings) == num_readings, \
                        "num of readings do not match! actual: %s, expected: %s" % (len(readings), num_readings)
                    position = data[(1 + num_readings): (1 + num_readings + 6)]
                    t = data[-1]
                    self.rear_dist.append([t] + position + readings)
                    self.rear_angles.append(angles)
                elif message_name == "ODOM":
                    position = data[0:6]
                    t = data[-1]
                    self.odom.append([t] + position)
                elif message_name == "#":
                    print("Parameter: %s" % line[1:])
                line = f.readline()
        self.front_dist_numpy, self.rear_dist_numpy, self.odom_numpy = \
            np.array(self.front_dist).astype(float), \
            np.array(self.rear_dist).astype(float), \
            np.array(self.odom).astype(float)
        self.front_angles_numpy, self.rear_angles_numpy = np.array(self.front_angles), np.array(self.rear_angles)

class LogSequencer():
    """
    Class for Sequencing a Carmen Log into multiple pieces.
    """
    def __init__(self, file_path, n_agent, laser_loc="FRONT", equal_length=True):
        self.loader = CarmenLoader(file_path)
        self.output_dir = path.join("/".join(file_path.split("/")[:-1]), "sequences")
        self.n_agent = n_agent
        self.laser_loc = laser_loc
        self.equal_length = equal_length
        print("Sequenced output directory: %s" % self.output_dir)

    def split(self):
        self.loader.load()
        n_steps = 0
        dist, angle = None, None
        if self.laser_loc == "FRONT":
            dist, angle = self.loader.front_dist_numpy, self.loader.front_angles_numpy
            n_steps = dist.shape[0]
        elif self.laser_loc == "REAR":
            dist, angle = self.loader.rear_dist_numpy, self.loader.rear_angles_numpy
            n_steps = dist.shape[0]

        k = n_steps // self.n_agent
        print("Total number of steps in the sequence: %d; Number of steps per agent: %d" % (n_steps, k))

        # create directory
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        for i in range(self.n_agent):
            start, end = k * i, k * (i + 1)
            dist_i = dist[start:end, :]
            output_path = path.join(self.output_dir, "seq_%i.csv" % i)
            np.savetxt(output_path, dist_i, delimiter=",")
            print("saved seq %d at: %s; Number of steps: %s" % (i, output_path, dist_i.shape[0]))

class SequenceLoader():

    def __init__(self, file_dir, seq):
        self.file_dir = file_dir
        self.seq = seq
        self.dist_numpy = None
        self.seq_length = None
        self.pointer = None
        self.num_laser = None

    def load(self):
        file_path = os.path.join(self.file_dir, "seq_%s.csv" % self.seq)
        self.dist_numpy = np.loadtxt(file_path, delimiter=',')
        self.seq_length = self.dist_numpy.shape[0]
        self.num_laser = self.dist_numpy.shape[1] - 7 # the first 7 data points are the robot positions.
        self.pointer = 0 # starting from the next array
        print("Loaded sequence %i; Number of timestamps: %s" % (self.seq, self.seq_length))

    def get_next_observations(self):
        # read the next {count} observations
        obs = self.dist_numpy[self.pointer, :]
        angles = np.array([-np.pi / 2 + i * np.pi / self.num_laser for i in range(self.num_laser)])

        self.pointer += 1
        # angle, dist, pos
        return angles, obs[1:4], obs[7:]

    def has_next(self):
        return self.pointer < self.seq_length - 1

if __name__=="__main__":
    file_path = "/home/jamesdi1993/datasets/2dlaser/orebro/orebro.gfs.log"
    sequencer = LogSequencer(file_path, 5)
    sequencer.split()