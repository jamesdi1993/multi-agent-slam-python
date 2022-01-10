from distributed_sparse_gp.agent import TSDFAgent
from distributed_sparse_gp.data_util import CarmenLoader
from distributed_sparse_gp.map_util import plot_agent_tsdf
from distributed_sparse_gp.perf import Timer
from matplotlib import pyplot as plt
import numpy as np

def run_dataset(loader):
    grid_min = np.array([-15, -25])
    grid_max = np.array([20, 10])

    width = grid_max[0] - grid_min[0]
    height = grid_max[1] - grid_min[1]

    sigma = 0.1
    mu_prior = 5.0
    c = 1.0
    l = 0.1
    max_leaf_size = 200
    truncated_dist = 0.5
    outlier_thresh = 20.0
    origin = np.array([0, 0, 0])
    intrinsic = [np.pi / 180, -np.pi / 2, np.pi / 2]
    grid_size = 0.1
    window_update = 5 # update every x steps

    agent = TSDFAgent(origin, grid_size, grid_min, grid_max, outlier_thresh,
                      c, l, sigma, mu_prior, truncated_dist, max_leaf_size, window_update)
    # agent = Agent(grid_min, grid_max, c, l, sigma, mu_prior)

    # t = loader.front_angles_numpy.shape[0]
    t = 100
    xx, yy = np.meshgrid(np.arange(grid_min[0], grid_max[0], grid_size), np.arange(grid_min[1], grid_max[1], grid_size))
    num_row, num_col = xx.shape[0], xx.shape[1]
    print("Number of rows: %s; col: %s" % (num_row, num_col))

    query_points = np.hstack((xx.reshape(-1, 1, order='C'), yy.reshape(-1, 1, order='C')))

    timer = Timer()
    for i in range(t):
        print("====================================")
        print("update at the %dth timestamp..." % i)
        # if i % 50 == 0 and i != 0:
        # print("update at the %dth timestamp..." % i)
        # occ_map.plot(loader.front_dist_numpy[:i, 1:4])
        pos = loader.front_dist_numpy[i, 1:4]  # (x, y, theta) of the robot position
        front_ang, front_dist = loader.front_angles_numpy[i, :], loader.front_dist_numpy[i, 7:]

        timer.start()
        agent.observe(front_dist, front_ang, pos)
        timer.stop()
        print("Time elapsed for converting pseudo-point: %s" % timer._last_elapsed_time)
        print("The pos of the robot is: %s" % pos)

    #== plot tsdf ==#
    plot_agent_tsdf(agent, query_points, grid_min, grid_max, grid_size, num_row, num_col, truncated_dist)

if __name__=="__main__":
    file_path = "/home/jamesdi1993/datasets/2dlaser/intel/intel.gfs.log"
    loader = CarmenLoader(file_path)
    loader.load()
    print(loader.front_dist_numpy)
    print(loader.front_dist_numpy.shape, )
    print(loader.rear_dist_numpy.shape, )

    #== run dataset ==#
    run_dataset(loader)