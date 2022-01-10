from agent import TimeVaryingDistributedAgent, CentralizedAgent
from commons import OUTPUT_FILE_FORMAT
from data_util import SequenceLoader, load_params
from map_util import plot_agents_tsdf
from perf import TSDFEvaluator, TimeCollector, DegreeCollector, Timer
from numpy.linalg import norm
from os import path

import datetime
import numpy as np
import os

def get_network_graph(robot_pos, dist_thresh):
    """
    Return a graph G, that denotes the connectivity of the robots given their positions.
    :param robot_pos: robot's positions. A n x 2 matrix.
    :return: A nxn matrix.
    """
    n_agents = robot_pos.shape[0]
    A = np.zeros((n_agents, n_agents))
    for i in range(n_agents):
        for j in range(n_agents):
            if norm(robot_pos[i, :] - robot_pos[j, :]) < dist_thresh:
                A[i, j] = 1
                A[j, i] = 1
    return A

def run_distributed_time_varying(params):
    #== dataset config ==#
    data_config = params['data']
    dataset_name = data_config['name']
    dataset_path = data_config['path']
    mode = data_config['mode']
    T = 0
    if mode == 'custom':
        T = data_config['steps']

    #== map config ==#
    map_config = params['map']
    grid_min = map_config['grid_min']
    grid_max = map_config['grid_max']
    width = grid_max[0] - grid_min[0]
    height = grid_max[1] - grid_min[1]

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
    count_thresh = process_config['count_thresh']

    #== agent config ==#
    agent_config = params['agent']
    enable_communicate = agent_config['enable_communicate']
    n_agents = agent_config['n_agents']
    dist_thresh = agent_config['dist_thresh']
    central_robot_index = -1

    # == visualization config ==#
    visualization_config = params['visualization']
    plot_agent_indices = visualization_config['plot_agent_indices']
    window_plot = visualization_config['window_plot']

    # == evaluation config ==#
    evaluation_config = params['evaluation']
    window_evaluate = evaluation_config['window_evaluate']  # evaluate every y steps
    write_prediction = evaluation_config['write_prediction']
    output_dir = evaluation_config['output_dir']
    grid_size_eval = evaluation_config['grid_size_eval']

    now = datetime.datetime.now()
    output_dir = path.join(output_dir, OUTPUT_FILE_FORMAT
                           .format(n_agents, now.year, now.month, now.day, now.hour, dist_thresh))
    if not path.exists(output_dir):
        os.makedirs(output_dir)

    #== initialize agents and data loader ==#
    agents_array = []
    data_loaders = []
    min_seq_length, max_seq_length = 0, 0
    for i in range(n_agents):

        agent = TimeVaryingDistributedAgent(origin, grid_size, grid_min, grid_max, outlier_thresh,
                                                 c, l, sigma, mu_prior, truncated_dist, max_leaf_size, window_update,
                                                 n_agents, i, central_robot_index, count_thresh)

        agents_array.append(agent)
        loader = SequenceLoader(dataset_path, i)
        loader.load()
        max_seq_length = max(max_seq_length, loader.seq_length)
        min_seq_length = min(min_seq_length, loader.seq_length)
        data_loaders.append(loader)

    central_agent = CentralizedAgent(origin, grid_size, grid_min, grid_max, outlier_thresh,
                                     c, l, sigma, mu_prior, truncated_dist, max_leaf_size, window_update,
                                     n_agents, central_robot_index, count_thresh)


    #== pass other agents to each agent ==#
    for i in range(n_agents):
        agents = {}
        for j in range(n_agents):
            if i != j:
                agents[j] = agents_array[j]
        agents_array[i].set_agents(agents)

    #== time config ==#
    if mode == 'max':
        T = max_seq_length
    elif mode == 'min':
        T = min_seq_length

    #== query points ==#
    xx, yy = np.meshgrid(np.arange(grid_min[0], grid_max[0], grid_size_eval),
                         np.arange(grid_min[1], grid_max[1], grid_size_eval))
    num_row, num_col = xx.shape[0], xx.shape[1]
    print("Number of rows: %s; col: %s" % (num_row, num_col))
    query_points = np.hstack((xx.reshape(-1, 1, order='C'), yy.reshape(-1, 1, order='C')))
    evaluator = TSDFEvaluator(central_agent, agents_array, query_points, truncated_distance=truncated_dist,
                              window_evaluate=window_evaluate, write_prediction=write_prediction, output_dir=output_dir)

    time_collector = TimeCollector(n_agents, window_update)
    degree_collector = DegreeCollector(n_agents, window_update)

    timer = Timer()
    timer.start()
    ##== run observations at each timestamp
    for t in range(T):
        print("====================================")
        print("update at the %dth timestamp..." % t)
        robot_pos = np.zeros((n_agents, 3))

        # observe
        for j in range(n_agents):
            print("Loading observations for the %d agent" % j)
            loader = data_loaders[j]
            agent = agents_array[j]

            if loader.has_next():
                angle, pos, dist = loader.get_next_observations()
                agent.observe(dist, angle, pos, t)
                robot_pos[j, :] = pos

        A = get_network_graph(robot_pos, dist_thresh)
        degree_collector.collect_one_step(A)

        # send batch
        for j in range(n_agents):
            agent = agents_array[j]
            # print("updating map...")
            if enable_communicate:
                agent.send_batch(A, t) # send out the information

            agent.send_local_batch(central_agent, t)  # send the local_batch to central_agent
            # agent.update_observations_from_all_robots() # update with the local information

        # collect time and statistics
        for j in range(n_agents):
            agent = agents_array[j]
            time_collector.collect_one_step(agent, t)

        if t % window_evaluate == 0 and t != 0:
            timer.stop()
            print("Time elapsed for processing %s for %d agents: %s" %
                  (window_evaluate, n_agents, timer._last_elapsed_time))

            # plot_agents_tsdf(agents_array, query_points, grid_min, grid_max, grid_size, num_row, num_col,
            #                  truncated_dist, count_thresh, plot_title)
            timer.start()
            evaluator.evaluate_one_step_agents_with_time(t)

    # total time
    timer.stop()
    print("Total time elapsed for processing %s" % timer._total_elapsed_time)

    # plot errors over time
    evaluator.plot_metrics()
    evaluator.output_final_metrics()

    # plot performance over time
    time_collector.plot_time()

    degree_collector.plot_degrees()
    degree_collector.output_metrics()

    if enable_communicate:
        title = ""
    else:
        title = dataset_name

    plot_agents_tsdf(agents_array, query_points, grid_min, grid_max, grid_size,
                     num_row, num_col, truncated_dist, count_thresh, title, plot_agent_indices)

if __name__=="__main__":
    param_path = "../../config/uni-bonn/csail.yaml"
    params = load_params(param_path)
    print(params)
    run_distributed_time_varying(params)

