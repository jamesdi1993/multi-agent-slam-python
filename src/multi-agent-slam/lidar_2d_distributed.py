from agent import InMemoryEcholessDistributedAgent
from data_util import SequenceLoader, load_params
from map_util import plot_agents_tsdf
from perf import TSDFEvaluator, TimeCollector

import numpy as np
import yaml

def run_distributed_dataset(params):
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
    window_evaluate = process_config['window_evaluate'] # evaluate every y steps
    count_thresh = process_config['count_thresh']

    #== agent config ==#
    agent_config = params['agent']
    enable_communicate = agent_config['enable_communicate']
    n_agents = agent_config['n_agents']
    W = np.array(agent_config['weight'])

    if not enable_communicate:
        W = np.eye(n_agents)
    print("The weight matrix is: %s" % W)

    #== initialize agents and data loader ==#
    agents_array = []
    data_loaders = []
    min_seq_length, max_seq_length = 0, 0
    for i in range(n_agents):

        agent = InMemoryEcholessDistributedAgent(origin, grid_size, grid_min, grid_max, outlier_thresh,
                                                 c, l, sigma, mu_prior, truncated_dist, max_leaf_size, window_update,
                                                 W, n_agents, i, count_thresh)
        agents_array.append(agent)

        loader = SequenceLoader(dataset_path, i)
        loader.load()
        data_loaders.append(loader)
        max_seq_length = max(max_seq_length, loader.seq_length)
        min_seq_length = min(min_seq_length, loader.seq_length)

    central_agent = InMemoryEcholessDistributedAgent(origin, grid_size, grid_min, grid_max, outlier_thresh,
                                             c, l, sigma, mu_prior, truncated_dist, max_leaf_size, window_update,
                                             W, n_agents, -1, count_thresh)

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
    xx, yy = np.meshgrid(np.arange(grid_min[0], grid_max[0], grid_size), np.arange(grid_min[1], grid_max[1], grid_size))
    num_row, num_col = xx.shape[0], xx.shape[1]
    print("Number of rows: %s; col: %s" % (num_row, num_col))
    query_points = np.hstack((xx.reshape(-1, 1, order='C'), yy.reshape(-1, 1, order='C')))
    evaluator = TSDFEvaluator(central_agent, agents_array, query_points, truncated_distance=truncated_dist,
                              window_evaluate=window_evaluate)
    time_collector = TimeCollector(n_agents, window_update)

    ##== run observations at each timestamp
    for t in range(T):
        print("====================================")
        print("update at the %dth timestamp..." % t)

        # observe
        for j in range(n_agents):
            print("Loading observations for the %d agent" % j)
            loader = data_loaders[j]
            agent = agents_array[j]

            if loader.has_next():
                angle, pos, dist = loader.get_next_observations()
                agent.observe(dist, angle, pos, t)

                # print("updating map...")
                if enable_communicate:
                    agent.send_batch(t) # send out the information

                agent.send_local_batch(central_agent, t)  # send the local_batch to central_agent
                # agent.update_observations_from_all_robots() # update with the local information
                time_collector.collect_one_step(agent)

        # update all agents
        if t % window_update == 0:
            for j in range(n_agents):
                agent = agents_array[j]
                agent.update_observations_from_all_robots(t)  # update with the local information
                agent.expire_messages(t)
            print("Updating central agent at timestamp: %d" % (t,))
            central_agent.update_observations_from_all_robots(t)
            central_agent.expire_all()

        # evaluate
        evaluator.evaluate_one_step_agents_with_time(t)

    # plot errors over time
    evaluator.plot_metrics()
    # plot performance over time
    time_collector.plot_time()
    # plot tsdf evaluation over time
    if enable_communicate:
        title = ""
    else:
        title = dataset_name
    plot_agents_tsdf(agents_array, query_points, grid_min, grid_max, grid_size,
                     num_row, num_col, truncated_dist, count_thresh, title)

if __name__=="__main__":
    param_path = "../../config/uni-bonn/csail.yaml"
    params = load_params(param_path)
    print(params)
    run_distributed_dataset(params)

