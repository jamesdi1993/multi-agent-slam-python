from agent import InMemoryEcholessDistributedAgent
from map_util import plot_agents_tsdf
from perf import TSDFEvaluator, TimeCollector, Timer
from nclt_sequencer import NCLTSequenceCSVLoader
# from data_loader import NCLTSequenceLoader

import numpy as np
import yaml

def load_params(param_path):
    with open(param_path, 'r') as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return params

def run_distributed_dataset(params):
    #== dataset config ==#
    data_config = params['data']
    dates = data_config['dates']
    dataset_path = data_config['path']

    mode = data_config['mode']
    T = 0
    if mode == 'custom':
        T = data_config['steps']
    seq_num = 2

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
    down_sampling = process_config['down_sampling']
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
        date = dates[i]
        agent = InMemoryEcholessDistributedAgent(origin, grid_size, grid_min, grid_max, outlier_thresh,
                                                 c, l, sigma, mu_prior, truncated_dist, max_leaf_size, window_update,
                                                 W, n_agents, i, count_thresh)

        print("The index for agent is: %s" % (agent.index,))

        agents_array.append(agent)

        # loader = NCLTSequenceLoader(dataset_path, date, down_sampling)
        loader = NCLTSequenceCSVLoader(dataset_path, date, seq_num, down_sampling)
        loader.load()
        max_seq_length = max(max_seq_length, loader.get_length())
        min_seq_length = min(min_seq_length, loader.get_length())
        data_loaders.append(loader)

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
        T = int(max_seq_length * down_sampling)
    elif mode == 'min':
        T = int(min_seq_length * down_sampling)

    print("Length of sequence: %s" % T)

    #== query points ==#
    xx, yy = np.meshgrid(np.arange(grid_min[0], grid_max[0], grid_size), np.arange(grid_min[1], grid_max[1], grid_size))
    num_row, num_col = xx.shape[0], xx.shape[1]
    print("Number of rows: %s; col: %s" % (num_row, num_col))
    query_points = np.hstack((xx.reshape(-1, 1, order='C'), yy.reshape(-1, 1, order='C')))
    evaluator = TSDFEvaluator(central_agent, agents_array, query_points, truncated_distance=truncated_dist,
                               window_evaluate=window_evaluate)
    time_collector = TimeCollector(n_agents, window_update)

    plot_title = "%d agents" % n_agents

    timer = Timer()
    timer.start()
    ##== run observations at each timestamp
    for t in range(T):
        print("====================================")
        print("update at the %dth timestamp..." % t)

        for j in range(n_agents):
            print("Loading observations for the %d agent" % j)
            loader = data_loaders[j]
            agent = agents_array[j]

            if loader.has_next():
                obs = loader.get_next()

            if obs is not None:
                original_t, dist, angle, pos = obs
                agent.observe(dist, angle, pos, t)
                print("The pos of the %dth robot at time %d is: %s" % (j, t, pos))

                # print("updating map...")
                if enable_communicate:
                    agent.send_batch(t) # send out the information

                agent.send_local_batch(central_agent, t) # send the local_batch to central_agent
                # agent.update_observations_from_all_robots() # update with the local information
            else:
                print("Reached the end of the dataset, breaking")
                break

        # update all agents
        if t % window_update == 0:
            for j in range(n_agents):
                agent = agents_array[j]
                agent.update_observations_from_all_robots(t)  # update with the local information
                agent.expire_messages(t)
                time_collector.collect_one_step(agent)

            print("Updating central agent at timestamp: %d" % (t,))
            central_agent.update_observations_from_all_robots(t)
            central_agent.expire_all()

        if t % window_evaluate == 0 and t != 0:
            timer.stop()
            print("Time elapsed for processing %s timestamp with %s down-sampling for %d agents: %s" %
                  (window_evaluate, down_sampling, n_agents, timer._last_elapsed_time))

            # plot_agents_tsdf(agents_array, query_points, grid_min, grid_max, grid_size, num_row, num_col,
            #                  truncated_dist, count_thresh, plot_title)
            timer.start()
            evaluator.evaluate_one_step_agents_with_time(t)

        # evaluator.evaluate_one_step_agents_with_time(t)

    # plot errors over time
    evaluator.plot_metrics()
    # plot performance over time
    time_collector.plot_time()
    # plot tsdf evaluation over time
    plot_agents_tsdf(agents_array, query_points, grid_min, grid_max, grid_size, num_row, num_col,
                     truncated_dist, count_thresh, plot_title)

if __name__=="__main__":
    param_path = "../../config/nclt/distributed_time_invariant.yaml"
    params = load_params(param_path)
    print(params)
    run_distributed_dataset(params)

