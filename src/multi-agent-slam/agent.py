from functools import reduce

from distributed_sparse_gp.map_util import aggregate_observations
from distributed_sparse_gp.quadtree import Node, GPModel
from distributed_sparse_gp.tsdf import TSDFHelper
from distributed_sparse_gp.perf import Timer
from message_manager import MessageManager

import numpy as np
import uuid

def simulate_stationary_distribution(W):
    """
    Simulate a stationary distribution.
    """
    v = 1.0 / W.shape[0] * np.ones(W.shape[0]).reshape(-1, 1) # col vector
    t = 1000
    for i in range(t):
        v_next = W @ v
        v = v_next

    v = v / np.sum(v)
    print("Stationary distribution shape: %s" % (v.shape,))
    return v

class Agent:

    def __init__(self, grid_min, grid_max, c, l, sigma, mu_prior, truncated_dist, max_leaf_size=200, count_thresh=0):
        # kernel params
        self.c = c
        self.l = l
        self.sigma = sigma

        # environment param
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.mu_prior = mu_prior
        self.truncated_dist = truncated_dist
        self.root = Node(grid_min[0], grid_min[1], grid_max[0] - grid_min[0], grid_max[1] - grid_min[1],
                        max_leaf_size, GPModel(self.c, self.l, self.sigma, self.mu_prior, self.truncated_dist,
                                               count_thresh, None, None, None, None, {}))

    def update(self, points, observations, counts):
        """
        Incrementally update the pseudo points with observations. Accounts for new pseudo-points.
        """
        self.root.insert(points, observations, counts, weights=np.ones(counts.shape))

    def predict(self, query_points, predict_cov=False):
        """
        predict the labels of the query points, according to pseudo-points
        """
        return self.root.evaluate(query_points, predict_cov)

    def get_points(self):
        return self.root.get_points()

    def get_num_pseudo_points(self):
        return self.root.get_num_pseudo_points()


class TSDFAgent(Agent):

    def __init__(self, origin, grid_size, grid_min, grid_max, outlier_thresh, c, l, sigma,
                 mu_prior, truncated_dist, max_leaf_size, window_update, count_thresh):
        super().__init__(grid_min, grid_max, c, l, sigma, mu_prior, truncated_dist, max_leaf_size, count_thresh)
        self.origin = origin
        self.grid_size = grid_size
        self.outlier_thresh = outlier_thresh

        self.window_update = window_update
        self.pseudo_points_cache = None
        self.tsdf_cache = None
        self.observation_count = 0
        self.tsdf_helper = TSDFHelper(origin, grid_size, grid_min, grid_max, truncated_dist, outlier_thresh)

    def observe(self, dist, angle, pos):
        pseudo_points, tsdf = self.tsdf_helper.transform(dist, angle, pos)

        if self.pseudo_points_cache is None:
            self.pseudo_points_cache = pseudo_points
            self.tsdf_cache = tsdf
        else:
            self.pseudo_points_cache = np.vstack((self.pseudo_points_cache, pseudo_points))
            self.tsdf_cache = np.vstack((self.tsdf_cache.reshape(-1, 1), tsdf.reshape(-1, 1)))

        self.observation_count += 1
        if self.observation_count >= self.window_update and self.pseudo_points_cache is not None \
                and self.pseudo_points_cache.shape[0] > 0:
            points_agg, tsdf_agg, counts = aggregate_observations(self.pseudo_points_cache, self.tsdf_cache)
            self.update(points_agg, tsdf_agg, counts)
            self.reset_cache()

    def reset_cache(self):
        self.pseudo_points_cache = None
        self.tsdf_cache = None
        self.observation_count = 0

'''
Abstraction for a distributed agent, that could either update its statistics from its neighbors,
or as a centralized agent (update from all agents)
'''
class AbstractDistributedAgent(TSDFAgent):

    def __init__(self, origin, grid_size, grid_min, grid_max, outlier_thresh,
                 c, l, sigma, mu_prior, truncated_dist, max_leaf_size, window_update,
                 W, n_agents, index, count_thresh):
        super().__init__(origin, grid_size, grid_min, grid_max, outlier_thresh, c, l, sigma,
                 mu_prior, truncated_dist, max_leaf_size, window_update, count_thresh)

        # for distributed estimation
        self.W = W  # weight matrix
        self.index = index  # the index of the current agent, for centralized agent, we use -1.
        self.n_agents = n_agents  # total number of agents
        self.received = [0 for i in range(n_agents)]

        # the robots statistics
        self.count_all_robots = {}
        self.observations_all_robots = {}
        self.pseudo_points_all_robots = {}

        # neighbors
        self.neighbors = self.W[self.index, :].nonzero()[0]
        print("The neighbors for agent %s: %s" % (self.index, self.neighbors))

    def receive_batch(self, agent_index, pseudo_points, observations, count):
        # receive and cache the observation received by a neighbor
        # print("Agent %s: Receiving observation from agent %s at clock %s" % (self.index, agent_index, cycle))
        if self.received[agent_index] == 1:
            raise Exception('Already updated statistics for neighbor: %i' % agent_index)

        # update neighbor parameter
        self.count_all_robots[agent_index] = count
        self.observations_all_robots[agent_index] = observations
        self.pseudo_points_all_robots[agent_index] = pseudo_points
        self.received[agent_index] = 1

        # print("received: %s" % received.nonzero()[0])
        # print("neighbors: %s" % self.neighbors)
        # print(np.all(received.nonzero()[0] == self.neighbors))

    # deprecated
    def receive(self, agent_index, points, obs, count):
        if self.received[agent_index] == 1:
            raise Exception('Already updated statistics for neighbor: %i' % agent_index)

        # update neighbor parameter
        self.count_all_robots[agent_index] = count
        self.observations_all_robots[agent_index] = obs
        self.pseudo_points_all_robots[agent_index] = points
        self.received[agent_index] = 1

    def get_observations(self, points, obs):
        raise NotImplementedError("Implement get observations for distributed agent!")

    # override
    def observe(self, dist, angle, pos):
        raise NotImplementedError("Implement distributed version of observe! Override base agent observe")


    def update_observations_from_all_robots(self):
        raise NotImplementedError("Implement update observations from all robots!")

    def reset_state(self):
        self.received = [0 for i in range(self.n_agents)]  # clear the states
        self.count_all_robots = {}
        self.observations_all_robots = {}
        self.pseudo_points_all_robots = {}

'''
A Distributed Agent. Corresponding to Section C in Journal Paper. 
'''
class DistributedAgent(AbstractDistributedAgent):

    def __init__(self, origin, grid_size, grid_min, grid_max, outlier_thresh,
                 c, l, sigma, mu_prior, truncated_dist, max_leaf_size, window_update,
                 W, n_agents, index, count_thresh):
        super().__init__(origin, grid_size, grid_min, grid_max, outlier_thresh,
                 c, l, sigma, mu_prior, truncated_dist, max_leaf_size, window_update,
                 W, n_agents, index, count_thresh)
        # the most recent observations
        self.new_pseudo = None
        self.new_obs = None
        self.new_count = None

    def get_observations(self, points, obs):
        """
        Store the observations
        """
        new_points_local, new_obs_local, new_count_local = aggregate_observations(points, obs)
        self.count_all_robots[self.index] = new_count_local
        self.observations_all_robots[self.index] = new_obs_local
        self.pseudo_points_all_robots[self.index] = new_points_local

    # TODO: Implement for Section C Agent.
    def observe(self, dist, angle, pos):
        pass

    def update_observations_from_all_robots(self):
        # Move current_robot's observations into a cache.
        self.received[self.index] = 1
        received = np.array(self.received)

        # First check if all neighbor's messages have been received
        if (np.all(received.nonzero()[0] != self.neighbors)):
            assert ("Not all messages have been received from neighbors for agent %s; received: %s; neighbors: %s"
                    % (self.index, received, self.neighbors))

        # TODO: Discount the original pseudo-point statistics, already present in the current Octree.
        for i in set(range(self.n_agents)) - set([self.index]):
            pseudo_i = self.pseudo_points_all_robots.get(i)
            obs_i = self.observations_all_robots.get(i)
            count_i = self.count_all_robots.get(i)

            if count_i is not None:
                self.root.insert(pseudo_i, count_i, obs_i, self.W[self.index, i] * np.ones(count_i.shape))
        """
        print("****************************************")
        print("Updated pseudo-points for agent %s: %s" % (self.index, self.pseudo_points))
        print("Updated observations for agent %s: %s" % (self.index, self.obs))
        print("Updated count for agent %s: %s" % (self.index, self.count))
        print("Updated Z matrix for agent %s: %s" % (self.index, self.Z))
        """
        # reset all the states
        self.reset_state()

    def get_time_spent_one_step(self):
        return self.timer._last_elapsed_time

class PseudoMessage():
    def __init__(self, pseudo_points, count, obs, receivers, message_id, t):
        self.pseudo_points = pseudo_points
        self.obs = obs
        self.count = count
        self.receivers = receivers
        self.id = message_id
        self.t = t

class InMemoryEcholessDistributedAgent(AbstractDistributedAgent):
    """
    A distributed agent that holds reference to the other agents in memory for communication.
    """
    def __init__(self, origin, grid_size, grid_min, grid_max, outlier_thresh,
                 c, l, sigma, mu_prior, truncated_dist, max_leaf_size, window_update,
                 W, n_agents, index, count_thresh):
        super().__init__(origin, grid_size, grid_min, grid_max, outlier_thresh,
                 c, l, sigma, mu_prior, truncated_dist, max_leaf_size, window_update,
                 W, n_agents, index, count_thresh)

        self.agents = {}  # dictionary of other agent objects, only initialized after all agents are created.
        self.received_ids = {} # message ids to keep track which message have been observed.
        self.message_all_robots = {}  # dictionary of {received_timestamp: robot_messages_dict }

        # robot_messsage_dict: {robot_id: message_array}
        print("The neighbors for agent %s: %s" % (self.index, self.neighbors))

        # stationary distribution
        self.pi = simulate_stationary_distribution(W)
        print("The stataionary distribution for agent %d is: %s" % (self.index, self.pi))
        self.timer = Timer()

    def set_agents(self, agents):
        self.agents = agents

    def send_batch(self, t):
        """
        Send out the batch to neighbors.
        :return: N/A
        """
        # TODO: Manage the message to keep in memory
        previous_message = self.message_all_robots.get(t-1)
        if previous_message is not None:
            message_array = list(previous_message.values()) # message from previous round
        else:
            message_array = []

        current_messages = self.message_all_robots.get(t)
        if current_messages is not None:
            new_obs = current_messages.get(self.index)  # new observations
            if new_obs is not None:
                message_array.append(new_obs)

        if len(message_array) > 0:
            messages = reduce(lambda a, b: a + b, message_array)

            # Note: only send the message received before the current round, or robot's new observations.
            for message in messages:
                message.senders.append(self.index)
                for neighbor in set(self.neighbors) - set([self.index]):
                    if neighbor not in message.senders:
                        agent = self.agents.get(neighbor)
                        if agent is not None:
                            agent.receive_batch(message, t)

    def send_local_batch(self, agent, t):
        """
        Send the local messages to agent
        :param agent:
        :return:
        """
        messages_t = self.message_all_robots.get(t)
        if messages_t is not None:
            messages = messages_t.get(self.index)
            if messages is not None:
                for message in messages:
                    agent.receive_batch(message, t)

    def receive_batch(self, message, t):
        if message.id in self.received_ids:
            return
        else:
            self.received_ids[message.id] = 1

        originator = message.senders[0]

        current_messages = self.message_all_robots.get(t)
        if current_messages is not None:
            robot_messages = current_messages.get(originator)
            if robot_messages is not None:
                robot_messages.append(message)
            else:
                current_messages[originator] = [message]
        else:
            self.message_all_robots[t] = {originator: [message]}

    def observe(self, dist, angle, pos, t):
        self.observation_count += 1
        pseudo_points, tsdf = self.tsdf_helper.transform(dist, angle, pos)
        if pseudo_points.shape[0] == 0:
            return

        if self.pseudo_points_cache is None:
            self.pseudo_points_cache = pseudo_points
            self.tsdf_cache = tsdf
        else:
            self.pseudo_points_cache = np.vstack((self.pseudo_points_cache, pseudo_points))
            self.tsdf_cache = np.vstack((self.tsdf_cache.reshape(-1, 1), tsdf.reshape(-1, 1)))

        if self.observation_count >= self.window_update and self.pseudo_points_cache is not None:
            pseudo_points, observations, counts = aggregate_observations(self.pseudo_points_cache, self.tsdf_cache)
            message = PseudoMessage(pseudo_points, counts, observations, [self.index], uuid.uuid4(), t)
            current_messages = self.message_all_robots.get(t)

            if current_messages is not None:
                robot_messages = current_messages.get(self.index)
                if robot_messages is not None:
                    robot_messages.append(message)
                else:
                    current_messages[self.index] = [message]
                self.reset_cache()
            else:
                self.message_all_robots[t] = {self.index: [message]}
                self.reset_cache()

    def update_observations_from_all_robots(self, t):
        # Step 1: update with all robots information
        self.timer.start()
        message_current = self.message_all_robots.get(t)

        # get the current round of messages
        if message_current is not None:
            for i in set(range(self.n_agents)):
                message_array = message_current.get(i)
                if message_array is not None:
                    for message in message_array:
                        pseudo_i, count_i, obs_i = message.pseudo_points, message.count, message.obs
                        self.root.insert(pseudo_i, obs_i, count_i, self.pi[i] * np.ones(count_i.shape))

                        # print("The number of pseudo-points for agent %d is: %s" %
                        #       (self.index, self.root.get_num_pseudo_points()))
        self.timer.stop()

    def expire_messages(self, t):
        for i in range(1, self.window_update + 1):
            message_t = self.message_all_robots.get(t - i)
            if message_t is not None:
                self.message_all_robots.pop(t - i)

    def expire_all(self):
        self.message_all_robots = {}
        # TODO: expire messages from the received_ids

class TimeVaryingDistributedAgent(TSDFAgent):

    # TODO: Remove the W matrix, as it is not used here.
    def __init__(self, origin, grid_size, grid_min, grid_max, outlier_thresh,
                 c, l, sigma, mu_prior, truncated_dist, max_leaf_size, window_update,
                 n_agents, index, central_robot_index, count_thresh):
        super().__init__(origin, grid_size, grid_min, grid_max, outlier_thresh, c, l, sigma,
                         mu_prior, truncated_dist, max_leaf_size, window_update, count_thresh)

        self.index = index  # the index of the current agent, for centralized agent, we use -1.
        self.n_agents = n_agents  # total number of agents

        self.pi = (1.0 / n_agents) * np.ones((n_agents, 1))
        self.message_manager = MessageManager(n_agents, index, central_robot_index)
        self.last_sent_batch_t = -1
        self.agents = None
        self.timer = Timer()

    def set_agents(self, agents):
        self.agents = agents

    def send_batch(self, A, t):
        """
        Send out the batch to neighbors.
        :return: N/A
        """
        #== get neighbors ==#
        non_zeros = np.nonzero(A[self.index, :])
        if non_zeros is not None:
            neighbors = non_zeros[0]
        else:
            raise RuntimeError("Should not have no non-zero entries for neighbors: %s" % A)

        for neighbor in set(neighbors) - set([self.index]):
            agent = self.agents.get(neighbor)
            assert agent is not None, "Neighboring agent cannot be None: %d" % (neighbor)
            messages = self.message_manager.retrieve_message(neighbor, t)
            if messages is not None and len(messages) > 0:
                for message in messages:
                    agent.receive_batch(self.index, message, t)
                # only update last sent batch if the robot
                self.last_sent_batch_t = t - 1

    def receive_batch(self, agent_index, message, t):
        if not self.message_manager.has_received(message):
            self.timer.start()
            #== Step 1: add message to the manager ==#
            self.message_manager.add_message(message, t)

            #== Step 2: update agent's parameter ==#
            pseudo_i, count_i, obs_i = message.pseudo_points, message.count, message.obs
            self.root.insert(pseudo_i, obs_i, count_i, self.pi[agent_index] * np.ones(count_i.shape))

            self.timer.stop()

    def observe(self, dist, angle, pos, t):
        self.observation_count += 1
        pseudo_points, tsdf = self.tsdf_helper.transform(dist, angle, pos)
        if pseudo_points.shape[0] == 0:
            # nothing to be updated
            return

        if self.pseudo_points_cache is None:
            self.pseudo_points_cache = pseudo_points
            self.tsdf_cache = tsdf
        else:
            # TODO: use python list instead of numpy vstack here.
            self.pseudo_points_cache = np.vstack((self.pseudo_points_cache, pseudo_points))
            self.tsdf_cache = np.vstack((self.tsdf_cache.reshape(-1, 1), tsdf.reshape(-1, 1)))

        if self.observation_count >= self.window_update and self.pseudo_points_cache is not None:
            pseudo_points, observations, counts = aggregate_observations(self.pseudo_points_cache, self.tsdf_cache)
            message = PseudoMessage(pseudo_points, counts, observations, [self.index], self.generate_message_id(t), t)

            # update message manager and pseudo-point statistics.
            self.message_manager.add_message(message, t)
            self.root.insert(pseudo_points, observations, counts, self.pi[self.index] * np.ones(counts.shape))
            self.reset_cache()

    def send_local_batch(self, central_agent, t):
        """
        Send the local messages to an agent.
        :param agent: an agent
        :param t: the timestamp
        :return: N/A
        """
        assert central_agent is not None, "agent cannot be None when sending local batch"
        messages = self.message_manager.retrieve_local_message(t)
        if messages is not None and len(messages) > 0:
            for message in messages:
                central_agent.receive_batch(self.index, message, t)

    def generate_message_id(self, t):
        return "robot{0}_{1}".format(self.index, t)

    def get_num_messages(self):
        return self.message_manager.get_num_messages()

    def get_num_leaves(self):
        return self.root.get_num_leaves()

class CentralizedAgent(TSDFAgent):

    def __init__(self, origin, grid_size, grid_min, grid_max, outlier_thresh,
                 c, l, sigma, mu_prior, truncated_dist, max_leaf_size, window_update,
                 n_agents, index, count_thresh):

        super().__init__(origin, grid_size, grid_min, grid_max, outlier_thresh, c, l, sigma,
                         mu_prior, truncated_dist, max_leaf_size, window_update, count_thresh)
        self.index = index  # the index of the current agent, for centralized agent, we use -1.
        self.n_agents = n_agents  # total number of agents
        self.pi = (1.0 / n_agents) * np.ones((n_agents, 1))
        self.timer = Timer()

    def receive_batch(self, agent_index, message, t):
        self.timer.start()
        # == update agent's parameter ==#
        pseudo_i, count_i, obs_i = message.pseudo_points, message.count, message.obs
        self.root.insert(pseudo_i, obs_i, count_i, self.pi[agent_index] * np.ones(count_i.shape))
        self.timer.stop()