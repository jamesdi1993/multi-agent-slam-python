from distributed_sparse_gp.map_util import transform_from_body_to_world_frame
from numpy import linalg as LA
import numpy as np

def filter_range(dist, ang, thresh):
    """
    Filter out distances that are greater or lower than a threshold. Return the filtered dist and angle.
    """
    lower, upper = thresh[0], thresh[1]
    bool_array = np.logical_and(dist < upper, dist > lower)
    dist_filtered = dist[bool_array]
    angle_filtered = ang[bool_array]
    return angle_filtered, dist_filtered

def calculate_coefficients(points, pos):
    """
    Calculate the normalized coefficients of the lines between laser beams.
    :param points: the laser beam endpoints, in world frame. n x 2 array
    :param pos: the robot position
    :return the normal parameters for the laser lines, pointing to the robot position. n x 3 array
    """
    points_rolled = np.roll(points, -1, axis=0)

    # print("The rolled points are: %s" % points_rolled)
    A = points_rolled[:, 1] - points[:, 1]  # delta_y
    B = points[:, 0] - points_rolled[:, 0]  # - delta_x
    C = -np.multiply(A, points[:, 0]) - np.multiply(B, points[:, 1])

    # calculate normal, pointing to pos's plane
    normal_sign = np.sign(pos[0] * A + pos[1] * B + C)

    norm = LA.norm(np.hstack((A.reshape(-1, 1), B.reshape(-1, 1))), axis=1)

    """
    norm_indices = np.where(norm == 0)[0]
    if norm_indices.size > 0:
        print("Encountered zero norms: %s" % norm_indices)
        print("A: %s \n B: %s; points: %s; pos: %s" %
              (A[norm_indices], B[norm_indices], points[norm_indices, :], pos))
    """
    # print("A shape: %s; norm shape: %s" % (A.shape, norm.shape))

    A = np.multiply(np.divide(A, norm), normal_sign)
    B = np.multiply(np.divide(B, norm), normal_sign)
    C = np.multiply(np.divide(C, norm), normal_sign)

    return np.hstack((A.reshape(-1, 1), B.reshape(-1, 1), C.reshape(-1, 1)))

def rayToPseudo(pos, points, normals, thresh, origin, grid_size, frame):
    """
    Construct the pseudo-points from the ray-endpoint cells.
    :param pos: the position of the robot
    :points: the endpoints of the laser beams, in world frame, n x 2 array
    :normals: the normal parameters of the line segments, n x 2 array. Each param i is the normal of the line
    between the i and the (i+1)th point.
    :thresh: the distance to truncate at.
    :origin: the origin of the world frame
    :frame: the frame to collect the pseudo-points at, in (n x 2)
    """
    origin = origin[0:2] # only need the (x,y) component
    point_cells = np.floor((points - origin) / grid_size) # the cell index corredponding to the ray endpoints
    coordinate_center = origin + point_cells * grid_size + 0.5 * grid_size # center of the cells, n x 2

    # local-frame around each coordinates
    coordinates = np.repeat(coordinate_center, frame.shape[0], axis=0) + np.tile(frame, [points.shape[0], 1])

    normals = np.repeat(normals, frame.shape[0], axis=0)

    """
    normals_rolled = np.roll(normals, 1, axis = 0)
    dist1 = np.multiply(normals[:, 0], coordinates[:, 0]) + \
            np.multiply(normals[:, 1], coordinates[:, 1]) + normals[:, 2]
    dist2 = np.multiply(normals_rolled[:, 0], coordinates[:, 0]) + \
            np.multiply(normals_rolled[:, 1], coordinates[:, 1]) + \
            normals_rolled[:, 2]
    dist = np.minimum(dist1, dist2)
    """
    dist = np.multiply(normals[:, 0], coordinates[:, 0]) + \
           np.multiply(normals[:, 1], coordinates[:, 1]) + normals[:, 2]

    dist = dist[:-frame.shape[0]]  # discard the last distance, as the normal estimate would not be accurate.
    coordinates = coordinates[:-frame.shape[0], :]
    tsdf_bool = np.abs(dist) < thresh

    # print("boolean shape: %s" % tsdf_bool.shape)
    # print("distance shape: %s" % dist.shape)

    tsdf = dist[tsdf_bool]
    pseudo_points = coordinates[tsdf_bool, :]
    return pseudo_points, tsdf

def obsToPseudo(laser_dist, angles, pos, thresh, origin, grid_size, frame):
    """
    :param laser_dist: distances of laser readings, (n,) array
    :param angles: the laser angles
    :param pos: the pose of the robot, in [x, y, \theta]
    :param thresh: the threshold of TSDF
    :param origin: the origin of the workspace
    :param grid_size: the size of the grid
    :param frame: the coordinates of the frame, zero_mean
    """
    """
    indices = np.nonzero(laser_dist == 0)
    print("Indices of zero laser readings: %s" % (indices,))
    """

    points_x = np.multiply(laser_dist, np.cos(angles)).reshape(-1, 1)
    points_y = np.multiply(laser_dist, np.sin(angles)).reshape(-1, 1)
    points_body = np.hstack((points_x, points_y))

    # print(points_body.shape)
    # positions of the ray endpoints in world frame
    points_world = transform_from_body_to_world_frame(pos, points_body)

    # calculate the normal vectors and distances
    normals = calculate_coefficients(points_world, pos)

    pseudo_points, tsdf = rayToPseudo(pos, points_world, normals, thresh, origin, grid_size, frame)
    return pseudo_points, tsdf

class TSDFHelper():

    def __init__(self, origin, grid_size, grid_min, grid_max, truncated_dist, outlier_thresh):
        self.origin = origin
        self.grid_size = grid_size
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.truncated_dist = truncated_dist
        self.outlier_thresh = outlier_thresh

    def transform(self, dist, angle, pos):
        angle_filtered, dist_filtered = filter_range(dist, angle, self.outlier_thresh)

        # 3 x 3 frame
        frame_x_ind, frame_y_ind = np.meshgrid(np.linspace(-1, 1, 3), np.linspace(-1, 1, 3))
        # frame_x_ind, frame_y_ind = np.meshgrid(np.linspace(0, 1, 1), np.linspace(0, 1, 1))
        frame_x_coord = frame_x_ind * self.grid_size
        frame_y_coord = frame_y_ind * self.grid_size
        frame = np.hstack((frame_x_coord.reshape(-1, 1), frame_y_coord.reshape(-1, 1)))
        pseudo_points, tsdf = obsToPseudo(dist_filtered, angle_filtered, pos, self.truncated_dist, self.origin,
                                          self.grid_size, frame)
        return pseudo_points, tsdf