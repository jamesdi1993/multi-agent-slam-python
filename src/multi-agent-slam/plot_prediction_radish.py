from commons import OUTPUT_FILE_FORMAT
from data_util import load_params
from map_util import plot_tsdf
from matplotlib.ticker import FuncFormatter, FormatStrFormatter
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from os import path
from perf import rmse, eps
from tabulate import tabulate

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pds
import seaborn

RESULT_DIRECTORY = "/home/jamesdi1993/workspace/Distributed-Sparse-GP/results/csail"

# colormap = plt.cm.cool

colormap = plt.cm.get_cmap("hsv")

color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']


def get_color(index, n_agent):
    colormap = plt.cm.get_cmap("hsv")
    step = 255 // n_agent
    return colormap(index * step)


def read_result(file_format, t):
    file_path = file_format.format(t)

    input = np.load(file_path)
    points = input['points']
    agents_pred = input['agents_pred']
    num_pseudo = input['num_pseudo']
    truncation = input['truncation']
    errors = input['errors']
    num_messages = input['num_messages']
    num_leaves = input['num_leaves']
    print(points.shape, agents_pred.shape, num_pseudo.shape, errors.shape, truncation)
    return points, agents_pred, num_pseudo, errors.flatten(), truncation, num_messages, num_leaves


def plot_metrics(num_agents, metrics, metric_name, window_evaluate, save_fig, output_dir, legend_loc=1):
    fig = plt.figure(figsize=(10, 10))
    # fig_num_pseudo = plt.figure(figsize=(10, 5))
    with seaborn.axes_style("darkgrid"):
        ax = fig.add_subplot(111)
        # ax_num_pseudo = fig_num_pseudo.add_subplot(111)

    # Plot error
    error_df = pds.DataFrame(metrics, columns=["Agent %d" % (i + 1) for i in range(num_agents)])
    print(error_df.head())
    print(metrics.shape, )
    error_df['Time'] = np.arange(0, error_df.shape[0] * window_evaluate, window_evaluate).astype(int)

    line_plot = seaborn.lineplot(x="Time", y=metric_name, hue="Agent", linewidth=4,
                                 ax=ax, data=pds.melt(error_df, "Time", var_name='Agent', value_name=metric_name))
    line_plot.set_yticklabels(line_plot.get_yticks(), size=20)
    line_plot.set_xticklabels(line_plot.get_xticks(), size=20)
    ax.legend([], [], frameon=False)
    # ax.set_title(metric_name, fontsize=18)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))

    ax.legend(markerscale=10, loc=legend_loc, prop={'size': 20})

    if metric_name == "Number of Pseudo":
        plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: int(y)))
    elif metric_name == "RMSE":
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter("%.2fm"))
    elif metric_name == "RMSE Recomputed":
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter("%.2fm"))
    elif metric_name == "Number of Messages":
        plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: int(y)))
    elif metric_name == "Number of Leaves":
        plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: int(y)))

    if save_fig:
        file_path = output_dir + "/{}.png".format(metric_name)
        plt.savefig(file_path)


def set_up_figure(grid_min, grid_max):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=True)
    # fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    ax.set_xlim(grid_min[0], grid_max[0])
    ax.set_ylim(grid_min[1], grid_max[1])
    # ax.set_xlim(-800, 150)
    # ax.set_ylim(-600, 200)

    # ax.set_xlabel('East (m)')
    # ax.set_ylabel('North (m)')
    return fig, ax


def set_up_tsdf_figure(grid_min, grid_max):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=True)

    ax.set_xlim(grid_min[0], grid_max[0])
    ax.set_ylim(grid_min[1], grid_max[1])

    # ax.set_xlabel("North (m)")
    # ax.set_ylabel("West (m)")

    return fig, ax


def plot_trajectory(ax, traj, agent_index, n_agents):
    ax.plot(-traj[:, 1], traj[:, 0], 'o',
            color=color_cycle[agent_index], ms=2, label='Agent %d' % (agent_index + 1))
    # ax.plot(-traj[:, 1], traj[:, 0], 'o', color=get_color(agent_index, n_agents), ms=2, label='Agent %d' % (agent_index + 1))
    plt.axis('off')
    # plt.show()


def plot_trajectories(ax, trajectories, n_agents):
    for i in range(n_agents):
        traj = trajectories[i]
        ax.plot(-traj[:, 1], traj[:, 0], 'o',
                color=color_cycle[i], ms=3, label='Agent %d' % (i + 1))
        # ax.plot(-traj[:, 1], traj[:, 0], 'o', color=get_color(i, n_agents), ms=3, label='Agent %d' % (i + 1))
        plt.axis('off')
    # ax.legend(markerscale=6, loc=4, prop={'size':20})


def plot_tsdf(ax, points, mean, truncated_dist,
                         grid_min, grid_max, num_row, num_col, cmap='gray'):
    zz = mean.reshape(num_row, num_col, order='C')
    img = ax.imshow(zz, cmap=cmap, interpolation='nearest', \
                    extent=[grid_min[0], grid_max[0], grid_min[1], grid_max[1]],
                    vmin=-truncated_dist + eps, vmax=truncated_dist - eps, origin='lower')
    ax.axis('off')


def plot_single_tsdf(fig, ax, points, mean, truncated_dist,
                     grid_min, grid_max, num_row, num_col, add_color_bar, add_scale_bar, cmap='gray'):
    zz = mean.reshape(num_row, num_col, order='C')
    img = ax.imshow(zz, cmap=cmap, interpolation='nearest',
                    extent=[grid_min[0], grid_max[0], grid_min[1], grid_max[1]],
                    vmin=-truncated_dist + eps, vmax=truncated_dist - eps, origin='lower')
    # zz = mean.reshape(num_row, num_col, order='C')
    # img = ax.imshow(zz, cmap='gray', interpolation='nearest', \
    #                extent=[grid_min[0], grid_max[0], grid_min[1], grid_max[1]],
    #                vmin=-truncated_dist, vmax=truncated_dist, origin='lower')
    if add_color_bar:
        # add colorbar
        # fig.subplots_adjust(right=0.8)
        cbar_tsdf = fig.add_axes([0.75, 0.2, 0.02, 0.65])
        cbar_tsdf.tick_params(labelsize=20)
        fig.colorbar(img, cax=cbar_tsdf)

    if add_scale_bar:
        # add scalebar
        fontprops = fm.FontProperties(size=32)
        scalebar = AnchoredSizeBar(ax.transData,
                                   500, '500m', 'lower left',
                                   pad=0.01,
                                   color='red',
                                   frameon=False,
                                   size_vertical=10,
                                   fontproperties=fontprops)
        ax.add_artist(scalebar)
    ax.axis('off')


# def plot_final_tsdf(agent_index, points, prediction, trajectories,
#                     truncated_dist, grid_min, grid_max, num_row, num_col):
#     # == Plot tsdf for each distributed agent ==#
#     n_agents = len(agent_index)
#     fig_tsdf, axes_tsdf = plt.subplots(1, n_agents, figsize=(5 * n_agents, 5))
#
#     for i, index in enumerate(agent_index):
#         agent_name = str(index + 1)  # zero to one-indexed
#
#         # == plot tsdf ==#
#         mean = prediction[index, :]
#         mean[mean > truncated_dist] = truncated_dist
#         mean[mean < -truncated_dist] = -truncated_dist
#
#         zz = mean.reshape(num_row, num_col, order='C')
#         img = plot_tsdf(axes_tsdf[i], zz, grid_min, grid_max, agent_name=agent_name,
#                         truncated_dist=truncated_dist)
#     # add colorbar
#     fig_tsdf.subplots_adjust(right=0.8)
#     cbar_tsdf = fig_tsdf.add_axes([0.825, 0.25, 0.05, 0.5])
#
#     fig_tsdf.colorbar(img, cax=cbar_tsdf)
#     # plt.tight_layout()
#     plt.show()

def main():
    save_fig = True

    # == parameters ==#
    param_path = "../../config/uni-bonn/csail.yaml"
    params = load_params(param_path)

    # == data config ==#
    data_config = params['data']
    # T = data_config['steps']
    T = 80
    dataset_path = data_config['path']

    # == map config ==#
    map_config = params['map']
    grid_min = map_config['grid_min']
    grid_max = map_config['grid_max']

    # == process config ==#
    process_config = params['process']
    truncated_dist = process_config['tsdf_thresh']
    down_sampling = process_config['down_sampling']

    # == agent config ==#
    agent_config = params['agent']
    num_agents = agent_config['n_agents']
    dist_thresh = agent_config['dist_thresh']

    # == evaluation config ==#
    evaluation_config = params['evaluation']
    window_evaluate = evaluation_config['window_evaluate']  # evaluate every y steps
    grid_size_eval = evaluation_config['grid_size_eval']

    # == run config ==#
    YY = 2021
    MM = 10
    DD = 10
    HH = 11
    num_hits = 180

    end = int(T / down_sampling)

    num_row = int((grid_max[1] - grid_min[1]) / grid_size_eval)
    num_col = int((grid_max[0] - grid_min[0]) / grid_size_eval)

    output_directory = path.join(RESULT_DIRECTORY, OUTPUT_FILE_FORMAT.format(num_agents, YY, MM, DD, HH, dist_thresh))
    file_format = output_directory + "/" + "eval_time{}.npz"
    errors_total, num_pseudo_total, num_messages_total, num_leaves_total = [], [], [], []

    # Read the centralized agent prediction first
    _, agents_pred, _, _, _, _, _ = read_result(file_format, T)
    central_pred = agents_pred[-1, :, :].flatten()

    print("TSDF thresh: %s" % (truncated_dist - eps))
    tsdf_within_boundary_index = np.nonzero(np.logical_and(central_pred < truncated_dist - eps,
                                                           central_pred > - truncated_dist + eps))[0]

    print("TSDF within boundary_index shape: %s" % tsdf_within_boundary_index.shape[0])
    central_pred_within_boundary = central_pred[tsdf_within_boundary_index]

    print("The shape of central prediction is: %s" % (central_pred.shape,))
    errors_recomputed = []

    for t in range(window_evaluate, T + window_evaluate, window_evaluate):
        points, agents_pred, num_pseudo, errors, truncation, num_messages, num_leaves = read_result(file_format, t)
        errors_total.append(errors)
        num_pseudo_total.append(num_pseudo)
        num_messages_total.append(num_messages)
        num_leaves_total.append(num_leaves)

        error_t = []
        for i in range(num_agents):
            err_i = rmse(agents_pred[i, :, :].flatten()[tsdf_within_boundary_index], central_pred_within_boundary)
            # print("The recomputed error shape for %dth agent: %s" % (i, err_i.shape,))
            error_t.append(err_i.flatten())

        errors_recomputed.append(np.array(error_t).flatten())

    errors_recomputed_arr = np.array(errors_recomputed)
    print("The recomputed error are: %s" % (errors_recomputed_arr))
    plot_metrics(num_agents, errors_recomputed_arr, 'RMSE Recomputed', window_evaluate, save_fig, output_directory)

    error_numpy = np.array(errors_total)
    num_pseudo_numpy = np.array(num_pseudo_total)
    num_messages_numpy = np.array(num_messages_total)
    num_leaves_numpy = np.array(num_leaves_total)

    plot_metrics(num_agents, error_numpy, 'RMSE', window_evaluate, save_fig, output_directory, 1)
    plot_metrics(num_agents, num_pseudo_numpy,
                 'Number of Pseudo', window_evaluate, save_fig, output_directory, 4)
    plot_metrics(num_agents, num_messages_numpy,
                 'Number of Messages', window_evaluate, save_fig, output_directory, 4)
    plot_metrics(num_agents, num_leaves_numpy,
                 'Number of Leaves', window_evaluate, save_fig, output_directory, 4)

    final_points, final_agents_pred, final_num_pseudo, final_errors, truncated_dist, \
    final_num_messages, final_num_leaves = read_result(file_format, t)

    # Tables for final metrics
    header = ["RMSE Mean", "RMSE Std",
              "Pseudo-points mean", "Pseudo-points std",
              "Number of messages mean", "Number of messages std",
              "Number of leaves mean", "Number of leaves std"]
    row = [np.mean(final_errors), np.std(final_errors),
           np.mean(final_num_pseudo), np.std(final_num_pseudo),
           np.mean(final_num_messages), np.std(final_num_messages),
           np.mean(final_num_leaves), np.std(final_num_leaves)]
    table = [header, row]
    print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))
    print(tabulate(table, headers='firstrow', tablefmt='latex'))


    fig_tsdf, ax_tsdf = set_up_figure(grid_min, grid_max)
    central_mean = final_agents_pred[-1, :]
    plot_tsdf(ax_tsdf, points, central_mean, truncated_dist,
                         grid_min, grid_max, num_row, num_col, 'gray')

    if save_fig:
        output_file = output_directory + "/agent_{0}.png".format("central")
        plt.savefig(output_file)

    # Figure for separate TSDF Evaluation
    agent_indices = [1,2]
    for i, agent_index in enumerate(agent_indices):
        mean = final_agents_pred[agent_index, :]
        print("Shape of prediction: %s" % mean)

        fig, ax = set_up_figure(grid_min, grid_max)

        add_color_bar = (i == len(agent_indices) - 1)
        # add_color_bar = False
        add_scale_bar = (i == 0)
        add_scale_bar = False
        plot_single_tsdf(fig, ax, points, mean, truncated_dist,
                         grid_min, grid_max, num_row, num_col, add_color_bar, add_scale_bar, 'gray')

        if save_fig:
            output_file = output_directory + "/agent{0}_tsdf.png".format(agent_index + 1)
            plt.savefig(output_file)
    plt.show()


if __name__ == "__main__":
    main()