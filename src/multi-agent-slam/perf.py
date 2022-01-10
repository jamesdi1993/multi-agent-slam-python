from tabulate import tabulate
import numpy as np
import matplotlib.pyplot as plt
import pandas as pds
import seaborn
import time

eps = 1e-3

def rmse(pred, gt):
    """"""
    assert pred.shape == gt.shape, "Prediction and ground-truth shape don't match! Pred shape: %s; GT shape: %s" \
                                   % (pred.shape, gt.shape)
    pred = pred.reshape(-1, 1)
    gt = gt.reshape(-1, 1)
    diff = pred - gt
    return np.sqrt(diff.T.dot(diff) / pred.shape[0])

def rrmse(pred, gt):
    assert pred.shape == gt.shape, "Prediction and ground-truth shape don't match! Pred shape: %s; GT shape: %s" \
                                   % (pred.shape, gt.shape)
    pred = pred.reshape(-1, 1)
    gt = gt.reshape(-1, 1)
    diff = pred - gt
    return np.sqrt(diff.T.dot(diff) / pred.shape[0])

class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""

class Timer:
    def __init__(self):
        self._start_time = None
        self._last_elapsed_time = 0
        self._total_elapsed_time = 0

    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        self._last_elapsed_time = elapsed_time
        self._total_elapsed_time += elapsed_time
        # print(f"Elapsed time: {elapsed_time:0.4f} seconds")

class TimeCollector():

    def __init__(self, n_agents, window_update):
        self.n_agents = n_agents
        self.time_one_step = [None for i in range(self.n_agents)]
        self.window_update = window_update

    def collect_one_step(self, agent, t):
        if t != 0:
            timer = agent.timer
            index = agent.index

            steps = t // self.window_update
            average = timer._total_elapsed_time / steps # delta per step
            if self.time_one_step[index] is None:
                self.time_one_step[index] = np.array([average])
            else:
                self.time_one_step[index] = np.hstack((self.time_one_step[index], average))

    def get_time(self):
        time = np.array(self.time_one_step).T # t x n
        print("The shape of time is: %s" % (time.shape,))
        return time

    def plot_time(self):
        fig_metrics = plt.figure(figsize=(5, 5))
        with seaborn.axes_style("darkgrid"):
            ax_time = fig_metrics.add_subplot()

        time_np = self.get_time()
        # Plot error
        time_df = pds.DataFrame(time_np, columns=["Agent %d" % (i + 1) for i in range(self.n_agents)])
        print(time_df.head())
        time_df['Time'] = np.arange(0, time_df.shape[0] * self.window_update, self.window_update).astype(int)

        seaborn.lineplot(x="Time", y="Seconds", hue="Agent",
                         ax=ax_time, data=pds.melt(time_df, "Time", var_name='Agent', value_name='Seconds'))
        ax_time.set_title("Time Spent Per Update")

class MessageCounter:

    def __init__(self):
        self.message_sent = 0
        self.message_received = 0

class TSDFEvaluator:

    def __init__(self, central_agent, agents, points, truncated_distance,
                 window_evaluate, error_metric='rmse', write_prediction=False, output_dir=""):
        self.central_agent = central_agent
        self.points = points # points to be evaluated
        self.agents = agents
        self.n_agents = len(agents)
        self.truncated_distance = truncated_distance
        self.window_evaluate = window_evaluate

        self.errors = None # timestamps x n_agents
        self.num_pseudo_points = None
        self.num_messages = None # number of messages
        self.num_leaves = None

        self.error_metric = error_metric
        self.write_prediction = write_prediction
        self.output_dir = output_dir
        if self.write_prediction:
            assert self.output_dir != "", "Output directory cannot be empty when writing prediction."

    # deprecated
    def evaluate_one_step(self, agent_index):
        agent = self.agents[agent_index]
        # Here we treat central agent as the ground truth.
        y = self.central_agent.predict(self.points, False)
        pred = agent.predict(self.points, False)

        if self.error_metric == 'rmse':
            error = rmse(pred, y)
        return error

    def evaluate_one_step_agents(self, t):
        if self.write_prediction:
            output_file = self.output_dir + "/eval_time%d.npz" % t

        y = self.central_agent.predict(self.points, False)
        agents_pred = []

        errors = []
        num_pseudo_points = []
        num_messages = []
        num_leaves = []

        tsdf_within_boundary_index = np.nonzero(np.logical_and(y < self.truncated_distance - eps,
                                                               y > -self.truncated_distance + eps))[0]

        print("TSDF within boundary_index shape: %s" % tsdf_within_boundary_index.shape[0])
        y_within_boundary = y[tsdf_within_boundary_index]

        print("TSDF within boundary: %s" % y_within_boundary)

        for agent in self.agents:
            pred = agent.predict(self.points, False)
            agents_pred.append(pred)

            pred_within_boundary = pred[tsdf_within_boundary_index]
            if self.error_metric == 'rmse':
                error = rmse(pred_within_boundary, y_within_boundary)
            errors.append(error)
            num_pseudo_points.append(agent.get_num_pseudo_points())
            num_messages.append(agent.get_num_messages())
            num_leaves.append(agent.get_num_leaves())

        agents_pred.append(y)

        agents_pred_array = np.array(agents_pred)
        num_pseudo_points_array = np.array(num_pseudo_points)
        num_messages_array = np.array(num_messages)
        num_leaves_array = np.array(num_leaves)

        error_array = np.array(errors)
        np.savez(output_file, points=self.points, agents_pred=agents_pred_array,
                 num_pseudo = num_pseudo_points_array,
                 errors = error_array,
                 num_messages = num_messages_array,
                 num_leaves = num_leaves_array,
                 truncation=self.truncated_distance - eps)

        print("The errors are: %s" % (errors,))
        return error_array.reshape(1, -1), np.array(num_pseudo_points).reshape(1, -1), \
               num_messages_array.reshape(1, -1), num_leaves_array.reshape(1, -1)

    def evaluate_one_step_agents_with_time(self, t):
        if t % self.window_evaluate == 0 and t != 0:
            if self.errors is None:
                errors, num_pseudo, num_messages, num_leaves = self.evaluate_one_step_agents(t)
                self.errors = errors
                self.num_pseudo_points = num_pseudo
                self.num_messages = num_messages
                self.num_leaves = num_leaves
            else:
                errors, num_pseudo, num_messages, num_leaves = self.evaluate_one_step_agents(t)
                self.errors = np.vstack((self.errors, errors))
                self.num_pseudo_points = np.vstack((self.num_pseudo_points, num_pseudo))
                self.num_messages = np.vstack((self.num_messages, num_messages))
                self.num_leaves = np.vstack((self.num_leaves, num_leaves))

    def get_errors(self):
        return self.errors

    def reset(self):
        self.errors = None

    def plot_metrics(self):
        fig_rmse = plt.figure(figsize=(10, 5))
        fig_num_pseudo = plt.figure(figsize=(10, 5))
        fig_num_messages = plt.figure(figsize=(10, 5))
        fig_num_leaves = plt.figure(figsize=(10, 5))
        with seaborn.axes_style("darkgrid"):
            ax_error = fig_rmse.add_subplot(111)
            ax_num_pseudo = fig_num_pseudo.add_subplot(111)
            ax_num_messages = fig_num_messages.add_subplot(111)
            ax_num_leaves = fig_num_leaves.add_subplot(111)

        #== Plot error ==#
        error_df = pds.DataFrame(self.errors, columns=["Agent %d" % (i + 1) for i in range(self.n_agents)])
        print(error_df.head())
        print(self.errors.shape, )
        error_df['Time'] = np.arange(0, error_df.shape[0] * self.window_evaluate, self.window_evaluate).astype(int)

        seaborn.lineplot(x="Time", y="RMSE", hue="Agent",
                         ax=ax_error, data=pds.melt(error_df, "Time", var_name='Agent', value_name='RMSE'))
        ax_error.set_title("RMSE")

        #== Plot Number of pseudo ==#
        num_pseudo_df = pds.DataFrame(self.num_pseudo_points, columns=["Agent %d" % (i + 1) for i in range(self.n_agents)])
        print(num_pseudo_df.head())
        print(num_pseudo_df.shape, )
        num_pseudo_df['Time'] = np.arange(0, num_pseudo_df.shape[0] * self.window_evaluate, self.window_evaluate).astype(int)
        seaborn.lineplot(x="Time", y="Num Pseudo", hue="Agent",
                         ax=ax_num_pseudo, data=pds.melt(num_pseudo_df, "Time", var_name='Agent', value_name='Num Pseudo'))
        ax_num_pseudo.set_title("Number of Pseudo Points")

        #== Plot Number of Messages ==#
        num_messages_df = pds.DataFrame(self.num_messages,
                                      columns=["Agent %d" % (i + 1) for i in range(self.n_agents)])
        print(num_messages_df.head())
        print(num_messages_df.shape, )
        num_messages_df['Time'] = np.arange(0, num_messages_df.shape[0] * self.window_evaluate,
                                          self.window_evaluate).astype(int)
        seaborn.lineplot(x="Time", y="Number of messages", hue="Agent",
                         ax=ax_num_messages,
                         data=pds.melt(num_messages_df, "Time", var_name='Agent', value_name='Number of messages'))
        ax_num_messages.set_title("Number of Messages")

        #== Plot Number of Leaves ==#
        num_leaves_df = pds.DataFrame(self.num_leaves,
                                        columns=["Agent %d" % (i + 1) for i in range(self.n_agents)])
        print(num_leaves_df.head())
        print(num_leaves_df.shape, )
        num_leaves_df['Time'] = np.arange(0, num_leaves_df.shape[0] * self.window_evaluate,
                                          self.window_evaluate).astype(int)
        seaborn.lineplot(x="Time", y="Number of Leaves", hue="Agent",
                         ax=ax_num_leaves,
                         data=pds.melt(num_leaves_df, "Time", var_name='Agent', value_name='Number of Leaves'))
        ax_num_leaves.set_title("Number of Leaves")
        # fig_metrics.suptitle("Metrics over Time")
        # plt.show()


    def output_final_metrics(self):
        final_errors = self.errors[-1, :]
        final_num_pseudo_points = self.num_pseudo_points[-1, :]
        final_num_messages = self.num_messages[-1, :]
        final_num_leaves = self.num_leaves[-1, :]

        header = ["RMSE Mean", "RMSE Std",
                  "Pseudo-points mean", "Pseudo-points std",
                  "Number of messages mean", "Number of messages std",
                  "Number of leaves mean", "Number of leaves std"]
        row = [np.mean(final_errors), np.std(final_errors),
               np.mean(final_num_pseudo_points), np.std(final_num_pseudo_points),
               np.mean(final_num_messages), np.std(final_num_messages),
               np.mean(final_num_leaves), np.std(final_num_leaves)]
        table = [header, row]
        print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))
        print(tabulate(table, headers='firstrow', tablefmt='latex'))


class DegreeCollector:

    def __init__(self, n_agents, window_update):
        self.n_agents = n_agents
        self.degrees = []
        self.window_update = window_update

    def collect_one_step(self, A):
        neighbors = np.sum(A, axis=1) - 1
        self.degrees.append(neighbors) # (n,) array

    def get_degrees(self):
        degrees = np.array(self.degrees)  # t x n
        print("The shape of degrees is: %s" % (degrees.shape,))
        return degrees

    def plot_degrees(self):
        fig_metrics = plt.figure(figsize=(5, 5))
        with seaborn.axes_style("darkgrid"):
            ax_degrees = fig_metrics.add_subplot()

        degree_npy = self.get_degrees()
        # Plot number of neighbors
        degree_df = pds.DataFrame(degree_npy, columns=["Agent %d" % (i + 1) for i in range(self.n_agents)])
        print(degree_df.head())
        degree_df['Time'] = np.arange(0, degree_df.shape[0] * self.window_update, self.window_update).astype(int)

        seaborn.lineplot(x="Time", y="Neighbors", hue="Agent",
                         ax=ax_degrees, data=pds.melt(degree_df, "Time", var_name='Agent', value_name='Neighbors'))
        ax_degrees.set_title("Number of neighbors per Update")

    def output_metrics(self):
        degree_npy = self.get_degrees()
        degree_avg = np.mean(degree_npy, axis=0) # avg over time

        header = ["Degree Mean", "Degree Std"]
        row = [np.mean(degree_avg), np.std(degree_avg)]
        table = [header, row]
        print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))
        print(tabulate(table, headers='firstrow', tablefmt='latex'))

