from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from scipy.spatial.distance import cdist

import math
import numpy as np
import time

def tic():
  t = time.time()
  return t

def toc(t):
  elapsed = time.time() - t
  return elapsed

def rbf_kernel(Xq, X, c, l):
    """
    Eq. 1) of SP-GP NIPs paper. Instead of following strictly eq.1, we use length_scale, which is the
    convention used in sk-learn.
    :param X: (n, d) array
    :param c: scale for the rbf function
    :param l: width of the kernel
    """
    dist = cdist(Xq, X, 'euclidean')
    # print(dist.shape)
    return c * np.exp(-0.5 * np.square(dist) / (l ** 2))

def aggregate_observations(points, obs):
    """
    Return the DISTINCT pseudo, obs, and count, given a batch of pseudo-points and their observations
    """
    distinct_points, distinct_obs, distinct_counts = points[0, :], obs[0].reshape(-1, 1), \
                                                     np.array([[1]])
    distinct_points_dict = {tuple(points[0, :]): 0}

    for i in range(1, points.shape[0]):
        pt = points[i, :]
        new_obs = obs[i]
        m = len(distinct_points_dict)
        if tuple(pt) in distinct_points_dict:
            index = distinct_points_dict[tuple(pt)]
            distinct_obs[index] = (distinct_obs[index] * distinct_counts[index] + new_obs) / (
                        distinct_counts[index] + 1)
            distinct_counts[index] += 1
        else:
            distinct_points = np.vstack((distinct_points, pt))
            distinct_obs = np.vstack((distinct_obs, new_obs))
            distinct_counts = np.vstack((distinct_counts, [1]))
            distinct_points_dict[tuple(pt)] = m
    return distinct_points, distinct_obs, distinct_counts

def plot_1d(query_points, mean, pseudo_points, obs):
    fig = plt.figure(figsize=(10, 5))
    plt.scatter(query_points.squeeze(), mean.squeeze(), label='pred', color='red', marker='*', s=5)
    plt.scatter(pseudo_points.squeeze(), obs.squeeze(), color='green', s=5, label='data')
    plt.legend(loc='lower right')
    plt.show()

def plot_2d(ax, query_points, mean_pred, pseudo_points, obs, title):
    ax.scatter3D(query_points[:, 0], query_points[:, 1], mean_pred.squeeze(), label='pred', color='red', marker='*', s=5)
    ax.scatter3D(pseudo_points[:, 0],  pseudo_points[:, 1], obs.squeeze(), color='green', s=5, label='data')
    # ax.plot_surface(query_points[:, 0], query_points[:, 1], mean_pred, label='pred', )
    # ax.plot_surface(pseudo_points[:, 0],  pseudo_points[:, 1], obs.reshape(-1, 1), label='data')

    ax.legend(loc='lower right')
    # ax.legend()
    ax.set_title(title)

def plot_agents_2D(n_agents, query_points, agents_pred, agents_pseudo, agents_obs, t):
    # fig, axes = plt.subplots(n_agents + 1, 1, figsize=(15, 20) ) # n_agents + central_agent
    fig = plt.figure()

    for i in range(n_agents):
        pred, pseudo, obs = agents_pred[i], agents_pseudo[i], agents_obs[i]
        ax = fig.add_subplot((n_agents + 1) / 2, 2, i + 1, projection='3d')
        title = "Agent %d" % i
        plot_2d(ax, query_points, pred, pseudo, obs, title)

    # plot central agent
    ax = fig.add_subplot((n_agents + 1) / 2 , 2, n_agents + 1, projection='3d')
    title = "Central Agent"
    plot_2d(ax, query_points, agents_pred[-1], agents_pseudo[-1], agents_obs[-1], title)
    fig.suptitle('timestamp %d' % t, fontsize=20)

def plot_error(errors, error_type='MSE'):
    fig = plt.figure(figsize=(15, 15))

    for i, err in enumerate(errors[:-1]):
        err_array = np.array(err)
        plt.plot(range(0, err_array.size), err_array, label="Agent %s" % i)

    plt.plot(errors[-1], label="Central agent")
    plt.legend()
    plt.title("%s for estimation" % error_type)
    plt.show()

def plot_tsdf(ax, prediction, grid_min, grid_max, agent_name, truncated_dist):
    img = ax.imshow(prediction, cmap='gray', interpolation='nearest', \
                              extent=[grid_min[0], grid_max[0], grid_min[1], grid_max[1]],
                              vmin= -truncated_dist, vmax=truncated_dist, origin='lower')
    # fig_tsdf.colorbar(img)
    # contour
    # contour = ax.contour(xx, yy, zz, levels=[0], colors='b')
    # Plot the pseudo-points value here
    ax.set_title("Agent %s" % agent_name)
    ax.set_xlim(grid_min[0], grid_max[0])
    ax.set_ylim(grid_min[1], grid_max[1])

    # ax.set_xlabel("North (m)")
    # ax.set_ylabel("West (m)")
    ax.axis('off')
    return img

def plot_agent_tsdf(agent, query_points, grid_min, grid_max, grid_size, num_row, num_col, truncated_dist, count_thresh):
    """
    Given the agents and query points, plot agent's tsdf.
    :param agents: The agent
    :param query_points: the points to evaluated at.
    :param grid_min: the minimum of the environment.
    :param: grid_max: the maximum of the environment.
    :param num_row: number of rows for each TSDF figure.
    :param num_col: number of columns for each TSDF Figure.
    :param truncated_dist: the truncated distance.
    """
    #== Plot tsdf for each distributed agent ==#
    fig_tsdf, ax_tsdf = plt.subplots(1, 1, figsize=(5, 5))
    fig_pseudo, ax_pseudo = plt.subplots(1, 1, figsize=(5, 5))
    agent_name = "Single Agent"

    #== plot tsdf ==#
    mean = agent.predict(query_points, False)
    zz = mean.reshape(num_row, num_col, order='C')
    img = plot_tsdf(ax_tsdf, zz, grid_min, grid_max, agent_name=agent_name,
                    truncated_dist=truncated_dist)

    #== plot pseudo-point==#
    pseudo_points, obs, counts = agent.get_points()
    sc = plot_pseudo_points(ax_pseudo, pseudo_points, obs, counts, grid_min, grid_max, grid_size,
                       agent_name=agent_name, thresh=truncated_dist, count_thresh=count_thresh)

    # add colorbar
    fig_tsdf.subplots_adjust(right=0.8)
    cbar_tsdf = fig_tsdf.add_axes([0.825, 0.25, 0.05, 0.5])

    fig_pseudo.subplots_adjust(right=0.8)
    cbar_pseudo = fig_pseudo.add_axes([0.825, 0.25, 0.05, 0.5])
    fig_tsdf.colorbar(img, cax=cbar_tsdf)

    fig_pseudo.colorbar(sc, cax=cbar_pseudo)

    fig_pseudo.suptitle("Pseudo-points", fontsize=18)
    plt.show()

def plot_agents_tsdf(agents, query_points, grid_min, grid_max, grid_size, num_row, num_col, truncated_dist,
                     count_thresh, title, agent_index):
    """
    Given the agents and query points, plot agent's tsdf.
    :param agents: The agents
    :param query_points: the points to evaluated at.
    :param grid_min: the minimum of the environment.
    :param: grid_max: the maximum of the environment.
    :param num_row: number of rows for each TSDF figure.
    :param num_col: number of columns for each TSDF Figure.
    :param truncated_dist: the truncated distance.
    :param title: title of the tsdf plots.
    """
    # == Plot tsdf for each distributed agent ==#
    n_agents = len(agent_index)
    fig_tsdf, axes_tsdf = plt.subplots(1, n_agents, figsize=(5 * n_agents, 5))
    fig_pseudo, axes_points = plt.subplots(1, n_agents, figsize=(5 * n_agents, 5))

    for i, index in enumerate(agent_index):
        agent = agents[index]
        agent_name = str(index + 1) # zero to one-indexed

        # == plot tsdf ==#
        mean = agent.predict(query_points, False)

        # mean[mean > truncated_dist] = truncated_dist
        # mean[mean < -truncated_dist] = -truncated_dist

        zz = mean.reshape(num_row, num_col, order='C')
        img = plot_tsdf(axes_tsdf[i], zz, grid_min, grid_max, agent_name=agent_name,
                        truncated_dist=truncated_dist)

        # == plot pseudo-point==#
        pseudo_points, obs, counts = agent.get_points()
        sc = plot_pseudo_points(axes_points[i], pseudo_points, obs, counts, grid_min, grid_max, grid_size,
                           agent_name=agent_name, thresh=truncated_dist, count_thresh=count_thresh)

    # add colorbar
    fig_tsdf.subplots_adjust(right=0.8)
    cbar_tsdf = fig_tsdf.add_axes([0.825, 0.25, 0.05, 0.5])

    fig_pseudo.subplots_adjust(right=0.8)
    cbar_pseudo = fig_pseudo.add_axes([0.825, 0.25, 0.05, 0.5])

    fig_tsdf.colorbar(img, cax=cbar_tsdf)
    fig_pseudo.colorbar(sc, cax=cbar_pseudo)
    if title != "":
        fig_tsdf.suptitle(title, fontsize=18)
    fig_pseudo.suptitle("Pseudo-points", fontsize=18)

    # plt.tight_layout()
    plt.show()

def plot_pseudo_points(ax, pseudo_points, obs, counts, grid_min, grid_max, grid_size, agent_name, thresh, count_thresh):

    indices = np.nonzero(counts > count_thresh)[0]
    sc = ax.scatter(pseudo_points[indices, 0], pseudo_points[indices, 1], c=obs[indices],
               cmap='plasma', vmin=-3 * grid_size, vmax=3 * grid_size, s= 1.0 * grid_size)
    # ax.set_xlabel("North (m)")
    # ax.set_ylabel("West (m)")

    ax.set_xlim(grid_min[0], grid_max[0])
    ax.set_ylim(grid_min[1], grid_max[1])
    ax.set_title("Agent: %s" % agent_name)
    return sc

def bresenham2D(sx, sy, ex, ey):
  '''
  Bresenham's ray tracing algorithm in 2D.
  Inputs:
	  (sx, sy)	start point of ray
	  (ex, ey)	end point of ray
  '''
  sx = int(round(sx))
  sy = int(round(sy))
  ex = int(round(ex))
  ey = int(round(ey))
  dx = abs(ex-sx)
  dy = abs(ey-sy)
  steep = abs(dy)>abs(dx)
  if steep:
    dx,dy = dy,dx # swap 

  if dy == 0:
    q = np.zeros((dx+1, 1))
  else:
    q = np.append(0,np.greater_equal(np.diff(np.mod(np.arange( np.floor(dx/2), -dy*dx+np.floor(dx/2)-1,-dy),dx)),0))
  if steep:
    if sy <= ey:
      y = np.arange(sy,ey+1)
    else:
      y = np.arange(sy,ey-1,-1)
    if sx <= ex:
      x = sx + np.cumsum(q)
    else:
      x = sx - np.cumsum(q)
  else:
    if sx <= ex:
      x = np.arange(sx,ex+1)
    else:
      x = np.arange(sx,ex-1,-1)
    if sy <= ey:
      y = sy + np.cumsum(q)
    else:
      y = sy - np.cumsum(q)
  return np.vstack((x,y)).astype(int)


def xy_to_rc(x_min, y_min, x, y, res):
  rows = np.int16(np.round((y - y_min) / res))
  cols = np.int16(np.round((x - x_min) / res))
  return np.vstack((rows, cols))

def transform_from_body_to_world_frame(pos, coords):
  """
  Transform coordinates from the body to the world frame, specified by pos.
  :param pos: The position of the robot. 3 x 1 array
  :param coords: The coordinates to be transformed; n x 2 array
  :return: The coordinates in the world frame. n x 2 array
  """
  coords_hom = to_homogenuous(coords) # copy n x 3
  r_to_w_matrix = np.identity(pos.shape[0]) # 3 x 3 matrix
  # p
  r_to_w_matrix[0, -1] = pos[0]
  r_to_w_matrix[1, -1] = pos[1]
  # rotation in 2D
  c, s = math.cos(pos[-1]), math.sin(pos[-1])
  r_to_w_matrix[0:2, 0:2] = np.array([[c, -s], [s, c]])
  c_transformed_hom = r_to_w_matrix @ coords_hom.T # 3 x n matrix
  return from_homogenuous(c_transformed_hom.T)

def to_homogenuous(coords_euclid):
  """
  Transform a coordinate from euclidean to homogenuous coordinate.
  :param coords_euclid: euclidean coordinates.  A m x d array
  :return: homogenuous coordinate. A m x (d + 1) array
  """
  # print("The shape of the coordinates is: %s" % (coords_euclid.shape, ))
  return np.append(coords_euclid, np.ones((coords_euclid.shape[0], 1)), axis = 1);

def from_homogenuous(coords_hom):
  """
  Transform a homogenuous coordinate into an euclidean coordinate.
  :param coords_hom: the homogenuous coordinate; A n x (d + 1) array
  :return: An euclidean coordinate. A n x d array
  """
  return np.divide(coords_hom[:, :-1], coords_hom[:, -1:])