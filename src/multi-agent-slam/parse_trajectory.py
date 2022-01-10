from data_util import load_params
import numpy as np
import os.path as path

def parse_sequence(data_dir, seq_num):
    data_file = data_dir + "/seq_{0}.yaml".format(seq_num)
    print("The current data path is: %s" % data_file)
    params = load_params(data_file)
    n_agents = params["n_agents"]
    pos = None

    min_length = 10000000
    trajectories = []

    trajectories_min = []

    for i in range(n_agents):
        date = params["agent_%d" % i]["date"]
        seq_name = "{}_seq_{}.csv".format(date, seq_num)
        trajectory = np.loadtxt(path.join(data_dir, seq_name), delimiter=",")
        trajectory_i = trajectory[:, [0, 1082, 1083]]
        print("The shape of trajectory is: %s" % (trajectory_i.shape,))
        min_length = min(min_length, trajectory_i.shape[0])
        trajectories.append(trajectory_i)

    for i in range(n_agents):
        traj = np.expand_dims(trajectories[i][:min_length, :], axis=0)
        trajectories_min.append(traj)

    for i in range(n_agents):
        pos = np.concatenate(trajectories_min, axis=0)

    print("The position shape is: %s" % (pos.shape,))

    output = data_dir + "/trajectories_%s.npz" % (n_agents)
    np.savez(output, pos)


if __name__ == "__main__":
    data_dir = "/home/jamesdi1993/datasets/NCLT/sequences"
    seq_num = 2
    parse_sequence(data_dir, seq_num)
