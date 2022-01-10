from distributed_sparse_gp.agent import TSDFAgent
from distributed_sparse_gp.data_loader import NCLTSequenceLoader
from distributed_sparse_gp.map_util import plot_agent_tsdf
from distributed_sparse_gp.perf import Timer
from numpy.random import seed

import numpy as np
import os
import sys
import yaml

def load_params(param_path):
    with open(param_path, 'r') as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return params

def run_dataset(params):
    #== dataset config ==#
    data_config = params['data']
    date = data_config['date']
    dataset_path = data_config['path']

    t = 0
    if data_config['mode'] == "custom":
        t = data_config['steps']
    else:
        t = sys.maxsize

    #== map config ==#
    map_config = params['map']
    grid_min = map_config['grid_min']
    grid_max = map_config['grid_max']

    origin = np.array(map_config['origin'])
    grid_size = map_config['grid_size']
    # TODO: Make the Lidar config a parameter.
    # intrinsic = [np.pi / 180, -np.pi / 2, np.pi / 2]

    #== process config==#
    process_config = params['process']
    truncated_dist = process_config['tsdf_thresh']
    outlier_thresh = process_config['outlier_thresh']
    sigma = process_config['sigma']
    mu_prior = process_config['mu_prior']
    c = process_config['c']
    l = process_config['l']
    max_leaf_size = process_config['max_leaf_size']
    window_update = process_config['window_update']  # update every x steps
    window_evaluate = process_config['window_evaluate'] # evaluate every y steps
    down_sampling = process_config['down_sampling']
    count_thresh = process_config['count_thresh']

    #== configure seed==#
    seed(1)

    #== initialize agents and data loader ==#
    agent = TSDFAgent(origin, grid_size, grid_min, grid_max, outlier_thresh,
                      c, l, sigma, mu_prior, truncated_dist, max_leaf_size, window_update, count_thresh)

    #== query points ==#
    xx, yy = np.meshgrid(np.arange(grid_min[0], grid_max[0], grid_size), np.arange(grid_min[1], grid_max[1], grid_size))
    num_row, num_col = xx.shape[0], xx.shape[1]
    print("Number of rows: %s; col: %s" % (num_row, num_col))
    query_points = np.hstack((xx.reshape(-1, 1, order='C'), yy.reshape(-1, 1, order='C')))

    #== configure loader ==#
    # laser_path = os.path.join(dataset_path, date + "_hokuyo", date, "hokuyo_30m.bin")
    # laser_loader = Hokuyo30mLoader(laser_path, down_sampling)
    # gt_path = os.path.join(dataset_path, "groundtruth_" + date + ".csv")
    # cov_path = os.path.join(dataset_path, "cov_" + date + ".csv")
    # gt_loader = GroundTruthLoader(gt_path, cov_path)
    #
    # gt_loader.load()
    # laser_loader.load()

    data_loader = NCLTSequenceLoader(dataset_path, date, down_sampling)
    data_loader.load()

    timer = Timer()
    timer.start()
    for i in range(t):
        print("====================================")
        print("update at the %dth timestamp..." % i)
        obs = data_loader.get_next()
        if obs is not None:
            t, dist, angle, pos = obs
            agent.observe(dist, angle, pos)
            print("The pos of the robot is: %s" % pos)
        else:
            print("Reached the end of the dataset, breaking")
            break

        if i % window_evaluate == 0 and i != 0:
            timer.stop()
            print("Time elapsed for processing %s timestamp with %s down-sampling: %s" %
                  (window_evaluate, down_sampling, timer._last_elapsed_time))
            plot_agent_tsdf(agent, query_points, grid_min, grid_max, grid_size, num_row, num_col, truncated_dist,
                            count_thresh=count_thresh)
            timer.start()

    timer.stop()
    print("Total elapsed for processing: %s" %
          (timer._total_elapsed_time))
    plot_agent_tsdf(agent, query_points, grid_min, grid_max, grid_size, num_row, num_col, truncated_dist,
                    count_thresh=count_thresh)

if __name__=="__main__":
    param_path = "../../config/nclt/2012_03_17.yaml"
    params = load_params(param_path)
    print(params)
    run_dataset(params)

