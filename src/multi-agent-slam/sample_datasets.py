from agent import Agent, DistributedAgent
from map_util import plot_2d, plot_error, plot_agents_2D

import numpy as np
import os.path as path
import pandas as pds
import yaml

class Dataset:

    def __init__(self, dataset_name, sigma_data):
        self.dataset_prefix = "/home/jamesdi1993/datasets/synthetic"
        self.data_path = dataset_name + ".csv"
        self.config_path = dataset_name + ".yaml"
        self.data = None
        self.config = None
        self.sigma = sigma_data

    def read(self):
        data_path = path.join(self.dataset_prefix, self.data_path)
        config_path = path.join(self.dataset_prefix, self.config_path)

        print("Loading data from: %s" % data_path)
        print("Loading config from: %s" % config_path)
        self.data = pds.read_csv(data_path)

        print(self.data.columns.values)
        with open(config_path, 'r') as stream:
            try:
                self.config = yaml.safe_load(stream)
                print(self.config)
            except yaml.YAMLError as exc:
                print("Error encoutered when reading config file: \n" + exc)

    def add_noise(self):
        noise = np.random.normal(0, self.sigma, self.data['obs'].to_numpy().size)
        self.data['obs'] = self.data['obs'] + noise

    def get_features(self):
        pass

    def get_observations(self):
        return self.data['obs']

    def shuffle(self):
        if self.data is not None:
            self.data = self.data.sample(frac=1).reset_index(drop=True)
        else:
            print("Dataset %s not loaded, please read dataset first." % self.data_path)

class MixtureGaussian2D(Dataset):

    def __init__(self, sigma_data):
        super().__init__("2d/mixture_gaussians", sigma_data)

    def get_features(self):
        return self.data[['x', 'y']].to_numpy()

    def get_partitions_x(self, n_partitions=3, padding_percent = 0.2):
        """
        Get partitions on the x dimension, from the dataset
        :param n_partitions: the number of partitions
        :return: the partitions
        """
        partition_array = []
        xmin = self.config['grid_min'][0]
        xmax = self.config['grid_max'][0]
        partition_min = np.linspace(xmin, xmax, n_partitions, endpoint=False)
        stride = (xmax - xmin) / n_partitions
        padding = stride * padding_percent

        partition_max = partition_min + stride + padding
        partition_max[-1] = xmax
        partition_min[-1] = partition_min[-1] - padding
        print("The partition min are: %s" % partition_min)
        print("The partition max are: %s" % partition_max)
        print("The shape of data is: %s" % (self.data.shape, ))

        for i in range(n_partitions):
            partition_bool = np.logical_and(self.data['x'] >= partition_min[i], self.data['x'] < partition_max[i])

            # print("The shape of the partition bool is: %s" % (partition_bool.shape,))
            partition = self.data.loc[partition_bool, :]

            # print("The %s th partition is: %s" % (i, partition))
            print("The shape of the partition is: %s" % (partition.shape,))
            partition_array.append(partition)
        return partition_array


def fit_mixture_gaussian_2d_single():
    c = 1
    l_list = [0.01, 0.1, 1.0, 10.0]
    sigma_data = 0.01
    sigma_process = 0.01
    mu_prior = 0
    dataset = MixtureGaussian2D(sigma_data)
    dataset.read()
    dataset.add_noise()

    X = dataset.get_features()
    y = dataset.get_observations()

    for l in l_list:
        agent = Agent(dataset.config["grid_min"], dataset.config["grid_max"], c, l, sigma_process, mu_prior)
        agent.update(X, y)
        mean_pred, cov = agent.predict(X)

        title = "c: %s, l: %s, sigma_data: %s, sigma_process: %s" % (c, l, sigma_data, sigma_process)
        plot_2d(X, mean_pred, X, y, title)

def fit_mixture_gaussian_2d_multi_agent():
    c = 1
    l = 0.1
    sigma_data = 0.01
    sigma_process = 0.01
    mu_prior = 0
    n_agents = 3
    padding = 0.1
    batch_size = 20
    plot_t = 3
    W = np.array([[0.5, 0.4, 0.1], [0.25, 0.5, 0.25], [0.1, 0.4, 0.5]])

    #===construct data===
    dataset = MixtureGaussian2D(sigma_data)
    dataset.read()
    dataset.add_noise()
    dataset.shuffle()

    #===split data===
    partitions = dataset.get_partitions_x(n_agents, padding)
    max_num_data = max([p.shape[0] for p in partitions])
    num_batch = int(max_num_data / batch_size)
    print("maximum obs: %s; number of batch: %s" % (max_num_data, num_batch))

    #===initialize agents===
    agents = [DistributedAgent(dataset.config["grid_min"], dataset.config["grid_max"], c, l, sigma_process,
                               mu_prior, W, n_agents, index) for index in range(n_agents)]
    central_agent = Agent(dataset.config["grid_min"], dataset.config["grid_max"], c, l, sigma_process, mu_prior)

    #===query points===
    query_points = dataset.get_features()
    query_points_obs = dataset.get_observations().to_numpy().reshape(-1, 1)

    #===MSE===
    mse = [[] for i in range(n_agents + 1)]

    for i in range(num_batch):
        start, end = i * batch_size, (i + 1) * batch_size

        agents_pred = []
        for j in range(n_agents):
            partition = partitions[j]
            if start > partition.shape[0]:
                continue
            elif end > partition.shape[0]:
                end = partition.shape[0]

            print("%sth batch, start: %s, end: %s" % (i, start, end))

            X, obs = partition[['x', 'y']].to_numpy(), partition['obs'].to_numpy()
            X, obs = X[start:end, :], obs[start:end]

            # update central agent
            central_agent.update(X, obs)

            # local agent
            receiver = agents[j]
            receiver.update_observations_from_all_robots(X, obs)
            pred, cov = receiver.predict(query_points)

            dist = pred - query_points_obs
            err = np.dot(dist.T.flatten(), dist.flatten()) / (end - start)

            print("Error for the %sth agent is: %s" % (j, err))
            mse[j].append(err)
            agents_pred.append(pred)

            # neighbors
            for k in range(n_agents):
                if W[j, k] != 0.0 and k != j:
                    sender = agents[k]
                    sender.send(receiver)

        # prediction for central agent
        central_pred, central_cov = central_agent.predict(query_points)
        central_dist = central_pred - query_points_obs
        central_err = np.dot(central_dist.flatten(), central_dist.flatten()) / (end - start)

        mse[-1].append(central_err)
        agents_pred.append(central_pred)

        agents_pseudo = [agent.pseudo_points for agent in agents]
        agents_pseudo.append(central_agent.pseudo_points)
        agents_obs = [agent.obs for agent in agents]
        agents_obs.append(central_agent.obs)

        if i % plot_t == 0:
            print("Plotting at the %d th timestamp" % (i))
            plot_agents_2D(n_agents, query_points, agents_pred, agents_pseudo, agents_obs, i)

    # plot_errors
    plot_error(mse)

if __name__ == '__main__':
    # fit_mixture_gaussian_2d_single()
    fit_mixture_gaussian_2d_multi_agent()