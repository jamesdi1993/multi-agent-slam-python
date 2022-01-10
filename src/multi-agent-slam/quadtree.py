from numpy.linalg import pinv
from distributed_sparse_gp.perf import Timer
from distributed_sparse_gp.map_util import rbf_kernel
import numpy as np

class GPModel():

    def __init__(self, c, l, sigma, mu_prior, truncated_dist, count_thresh=0,
                 pseudo_points=None, count=None, obs=None, Z=None,
                 pseudo_map={}):
        # kernel params
        self.c = c
        self.l = l
        self.sigma = sigma
        self.mu_prior = mu_prior
        self.truncated_dist = truncated_dist
        self.count_thresh = count_thresh

        self.pseudo_points = pseudo_points  # initially none
        self.count = count
        self.obs = obs
        self.Z = Z  # precision matrix
        self.pseudo_map = pseudo_map  # dictionary for fast loop-up pseudo-points

    def get_num_points(self):
        if self.pseudo_points is None:
            return 0
        else:
            return self.pseudo_points.shape[0]

    def find_pseudo_points(self, x, y, w, h):
        bool_arr = np.logical_and(self.pseudo_points[:, 0] >= x, self.pseudo_points[:, 0] <= x + w)
        bool_arr = np.logical_and(bool_arr, self.pseudo_points[:, 1] >= y)
        bool_arr = np.logical_and(bool_arr, self.pseudo_points[:, 1] <= y + h)
        # print("The shape of bool array is: %s; Shape of Z matrix: %s" % (bool_arr.shape, self.Z.shape))
        return np.nonzero(bool_arr)[0]

    def get_sub_model(self, pts_index):
        """
        Get the sub-model correspdoning to the indices of the points.
        :param pts_index: the index of the points
        :return:
        """
        points = np.array(self.pseudo_points[pts_index, :])
        obs = np.array(self.obs[pts_index])
        count = np.array(self.count[pts_index])

        non_zero_ind = np.where(count == 0)[0]

        if non_zero_ind.size > 0:
            print("Non zero counts encountered... Points: %s; Obs: %s; Count: %s"
                  % (points[non_zero_ind, :], obs[non_zero_ind], count[non_zero_ind]))

        # print("Shape of GP Z matrix: %s" % (self.Z.shape,))
        # TODO: This part is not correct -- the information matrix Z has to be re-computed again.
        Z_inv = rbf_kernel(points, points, self.c, self.l) + self.sigma ** 2 * np.diag(1.0 / count)
        Z = pinv(Z_inv)

        pseudo_map = {}
        for i in range(points.shape[0]):
            pseudo_map[tuple(points[i, :])] = i
        return points, obs, count, Z, pseudo_map

    def update_batch(self, points, observations, count):
        """
        Incrementally update the pseudo points with observations. Accounts for new pseudo-points.
        """
        n_points = points.shape[0]

        # print("Updating for %d number of observations; number of pseudo-point set: %d"
        #       % (n_points, len(self.pseudo_map)))

        t_existing = Timer()
        t_new = Timer()
        t_z_update = Timer()

        for i in range(n_points):
            pt = points[i, :]
            m = len(self.pseudo_map)
            B = self.Z

            # existing pseudo-point
            if tuple(pt) in self.pseudo_map:

                t_existing.start()
                index = self.pseudo_map[tuple(pt)]

                # update Z
                e_l = np.zeros((m, 1))
                e_l[index, 0] = 1
                eps = self.sigma ** 2 * (1.0 / (self.count[index] + count[i]) - 1.0 / (self.count[index]))
                t_z_update.start()
                self.Z = B - (B @ e_l @ e_l.T @ B) / (1.0 / eps + e_l.T @ B @ e_l)
                t_z_update.stop()

                # update obs and count
                self.obs[index] = (self.obs[index] * self.count[index] + observations[i]) / (
                            self.count[index] + count[i])
                self.count[index] += count[i]
                t_existing.stop()
            # new point
            else:

                t_new.start()
                point = pt.reshape(1, -1)
                D = rbf_kernel(point, point, self.c, self.l) + self.sigma ** 2
                if m == 0:
                    self.Z = pinv(D)
                    self.obs = np.array([observations[i]])
                    self.count = np.array([count[i]])
                    self.pseudo_points = point
                else:
                    # print(point)
                    # print(self.pseudo_points)
                    C = rbf_kernel(self.pseudo_points, point, self.c, self.l)
                    D = rbf_kernel(point, point, self.c, self.l) + self.sigma ** 2

                    S = pinv(D - C.T @ B @ C)

                    # print("D: %s" % D)
                    # print("C: %s" % C)
                    # print("B: %s" % B)
                    # print("S: %s" % S)
                    # update Z matrix
                    V1 = np.hstack((B + B @ C @ S @ C.T @ B, - B @ C @ S))
                    V2 = np.hstack((- S @ C.T @ B, S))
                    self.Z = np.vstack((V1, V2))

                    self.pseudo_points = np.vstack((self.pseudo_points, point))
                    self.obs = np.append(self.obs, [observations[i]])
                    self.count = np.append(self.count, [count[i]])

                index = m
                self.pseudo_map[tuple(pt)] = index  # update index
                t_new.stop()

        print("The elapsed time for updating existing pseudo-point: %s" % t_existing._total_elapsed_time)
        print("The elapsed time for Z matrix update: %s" % t_z_update._total_elapsed_time)
        print("The elapsed time for new pseudo-point: %s" % t_new._total_elapsed_time)


    def update(self, points, observations, counts, weights):
        """
        Update the pseudo-point statistics based, with weights incorporating different robots.
        """
        n_points = points.shape[0]
        # print("Updating for %d number of observations; number of pseudo-point set: %d"
        #       % (n_points, len(self.pseudo_map)))

        for i in range(n_points):
            pt = points[i, :]
            m = len(self.pseudo_map)
            B = self.Z
            # existing pseudo-point
            if tuple(pt) in self.pseudo_map:
                index = self.pseudo_map[tuple(pt)]
                # update Z
                e_l = np.zeros((m, 1))
                e_l[index, 0] = 1
                eps = self.sigma ** 2 * (1.0 / (self.count[index] + counts[i]) - 1.0 / (self.count[index]))
                self.Z = B - (B @ e_l @ e_l.T @ B) / (1.0 / eps + e_l.T @ B @ e_l)
                # update obs and count
                self.obs[index] = (self.obs[index] * self.count[index] + observations[i] * weights[i] * counts[i]) \
                                  / (self.count[index] + weights[i] * counts[i])
                self.count[index] += weights[i] * counts[i]
            # new point
            else:
                point = pt.reshape(1, -1)
                D = rbf_kernel(point, point, self.c, self.l) + self.sigma ** 2
                if m == 0:
                    self.Z = pinv(D)
                    self.obs = np.array([observations[i]])
                    self.count = np.array([counts[i] * weights[i]])
                    self.pseudo_points = point
                else:
                    # print(point)
                    # print(self.pseudo_points.shape,)
                    C = rbf_kernel(self.pseudo_points, point, self.c, self.l)
                    D = rbf_kernel(point, point, self.c, self.l) + self.sigma ** 2

                    S = pinv(D - C.T @ B @ C)

                    # print("D: %s" % D)
                    # print("C: %s" % C)
                    # print("B: %s" % B)
                    # print("S: %s" % S)
                    # update Z matrix
                    V1 = np.hstack((B + B @ C @ S @ C.T @ B, - B @ C @ S))
                    V2 = np.hstack((- S @ C.T @ B, S))
                    self.Z = np.vstack((V1, V2))
                    self.pseudo_points = np.vstack((self.pseudo_points, point))
                    self.obs = np.append(self.obs, [observations[i]])
                    self.count = np.append(self.count, [counts[i] * weights[i]])
                index = m
                self.pseudo_map[tuple(pt)] = index  # update index

    def predict(self, query_points, predict_cov=True):
        """
        predict the labels of the query points, according to pseudo-points.
        """
        mu_0 = np.ones((query_points.shape[0], 1)) * self.mu_prior

        if self.count is not None:
            count_indices = np.nonzero(self.count.reshape(-1) > self.count_thresh)[0]

        # Empty Pseudo-point set. This could happen when: a) the beginning of the sequence;
        # b) when a parent node is split; c) non of the points have been observed more than once.
        if len(self.pseudo_map) == 0 or count_indices.size == 0:
            mean = mu_0
            if predict_cov:
                cov = rbf_kernel(query_points, query_points, self.c, self.l)
                return mean, cov
            else:
                return mean

        points = self.pseudo_points[count_indices, :]
        mu_P = np.ones((points.shape[0], 1)) * self.mu_prior
        obs = self.obs[count_indices]

        kxP = rbf_kernel(query_points, points, self.c, self.l)
        mean = mu_0 + kxP @ self.Z[count_indices, :][:, count_indices] @ (obs.reshape(-1, 1) - mu_P)

        # Truncate distance
        mean[mean > self.truncated_dist] = self.truncated_dist
        mean[mean < -self.truncated_dist] = -self.truncated_dist

        if predict_cov:
            kxx = rbf_kernel(query_points, query_points, self.c, self.l)
            cov = kxx - kxP @ self.Z[count_indices, count_indices] @ kxP.T
            return mean, cov
        return mean

class Node():

    def __init__(self, x0, y0, w, h, max_size, gp_model):
        self.x0 = x0
        self.y0 = y0
        self.width = w
        self.height = h
        self.max_size = max_size
        self.children = []
        # 1 | 2
        # -----
        # 0 | 3
        self.isLeaf = True
        self.model = gp_model

    def get_width(self):
        return self.width

    def get_height(self):
        return self.height

    def get_model(self):
        return self.model

    def get_points(self):
        children = find_children(self)
        points, obs, counts = [], [], []
        for leaf in children:
            if leaf.model.pseudo_points is not None:
                points.append(leaf.model.pseudo_points)
                obs.append(leaf.model.obs.reshape(-1, 1))
                counts.append(leaf.model.count.reshape(-1, 1))

                # print("Points shape: %s; obs shape: %s" % (leaf.model.pseudo_points.shape, leaf.model.obs.shape))

        print("The shape of points are: %s" % len(points))
        return np.vstack(points), np.vstack(obs).reshape(-1), np.vstack(counts).reshape(-1)

    def get_num_pseudo_points(self):
        if self.isLeaf:
            return self.get_model().get_num_points()
        else:
            num_points = [child.get_num_pseudo_points() for child in self.children]
            return sum(num_points)

    def get_num_leaves(self):
        if self.isLeaf:
            return 1
        else:
            num_leaves = [child.get_num_leaves() for child in self.children]
            return sum(num_leaves)

    def evaluate(self, query_points, predict_cov):
        if self.isLeaf:
            return self.model.predict(query_points, predict_cov)
        else:
            # TODO: Use array to simply this part of the code
            index_array_0 = contains_points(self.children[0].x0, self.children[0].y0, self.children[0].width, self.children[0].height, query_points);
            index_array_1 = contains_points(self.children[1].x0, self.children[1].y0, self.children[1].width, self.children[1].height, query_points);
            index_array_2 = contains_points(self.children[2].x0, self.children[2].y0, self.children[2].width, self.children[2].height, query_points);
            index_array_3 = contains_points(self.children[3].x0, self.children[3].y0, self.children[3].width, self.children[3].height, query_points);

            points0 = query_points[index_array_0, :].reshape(-1, 2)
            points1 = query_points[index_array_1, :].reshape(-1, 2)
            points2 = query_points[index_array_2, :].reshape(-1, 2)
            points3 = query_points[index_array_3, :].reshape(-1, 2)

            mean0 = self.children[0].evaluate(points0, False)
            mean1 = self.children[1].evaluate(points1, False)
            mean2 = self.children[2].evaluate(points2, False)
            mean3 = self.children[3].evaluate(points3, False)

            mean = np.zeros((query_points.shape[0], 1))
            mean[index_array_0] = mean0
            mean[index_array_1] = mean1
            mean[index_array_2] = mean2
            mean[index_array_3] = mean3
            # note that the order of the points are not maintained.
            # mean = np.vstack((mean0.reshape(-1, 1), mean1.reshape(-1, 1), mean2.reshape(-1, 1), mean3.reshape(-1, 1),))
            # points = np.vstack((points0, points1, points2, points3))
            # return mean, points
            return mean

    def insert(self, pseudo_points, obs, count, weights):
        if self.isLeaf:
            self.model.update(pseudo_points, obs, count, weights)
            # self.model.update_batch(pseudo_points, obs, count)
            # print("Z matrix shape: %s" % (self.model.Z.shape,))
            recursive_subdivide(self, self.max_size)

        else:
            bool_array_0 = contains_points(self.children[0].x0, self.children[0].y0, self.children[0].width,
                                           self.children[0].height, pseudo_points);
            bool_array_1 = contains_points(self.children[1].x0, self.children[1].y0, self.children[1].width,
                                           self.children[1].height, pseudo_points);
            bool_array_2 = contains_points(self.children[2].x0, self.children[2].y0, self.children[2].width,
                                           self.children[2].height, pseudo_points);
            bool_array_3 = contains_points(self.children[3].x0, self.children[3].y0, self.children[3].width,
                                           self.children[3].height, pseudo_points);
            # when splitting, the count and statistics already take into account the weights, and hence
            # they are initiating to be one here.
            self.children[0].insert(pseudo_points[bool_array_0], obs[bool_array_0], count[bool_array_0],
                                    weights[bool_array_0])
            self.children[1].insert(pseudo_points[bool_array_1], obs[bool_array_1], count[bool_array_1],
                                    weights[bool_array_1])
            self.children[2].insert(pseudo_points[bool_array_2], obs[bool_array_2], count[bool_array_2],
                                    weights[bool_array_2])
            self.children[3].insert(pseudo_points[bool_array_3], obs[bool_array_3], count[bool_array_3],
                                    weights[bool_array_3])

def get_sub_model(x, y, w, h, model):
    pts_index = model.find_pseudo_points(x, y, w, h)

    if pts_index.size > 0:
        points, obs, count, Z, pseudo_map = model.get_sub_model(pts_index)
        sub_model = GPModel(model.c, model.l, model.sigma, model.mu_prior, model.truncated_dist, model.count_thresh,
                            points, obs, count, Z, pseudo_map)
    else:
        sub_model = GPModel(model.c, model.l, model.sigma, model.mu_prior, model.truncated_dist, model.count_thresh,
                            None, None, None, None, {})
    return sub_model

def contains_points(x, y, w, h, points):
    bool_arr = np.logical_and(points[:, 0] >= x, points[:, 0] <= x + w)
    bool_arr = np.logical_and(bool_arr, points[:, 1] >= y)
    bool_arr = np.logical_and(bool_arr, points[:, 1] <= y + h)
    return np.nonzero(bool_arr)

def recursive_subdivide(node, k):
    num_points = node.model.get_num_points()
    if num_points < k:
        return

    w_ = float(node.width / 2)
    h_ = float(node.height / 2)

    m1 = get_sub_model(node.x0, node.y0, w_, h_, node.model)
    n1 = Node(node.x0, node.y0, w_, h_, node.max_size, m1)
    recursive_subdivide(n1, k)

    m2 = get_sub_model(node.x0, node.y0 + h_, w_, h_, node.model)
    n2 = Node(node.x0, node.y0 + h_, w_, h_, node.max_size, m2)
    recursive_subdivide(n2, k)

    m3 = get_sub_model(node.x0 + w_, node.y0, w_, h_, node.model)
    n3 = Node(node.x0 + w_, node.y0, w_, h_, node.max_size, m3)
    recursive_subdivide(n3, k)

    m4 = get_sub_model(node.x0 + w_, node.y0 + h_, w_, h_, node.model)
    n4 = Node(node.x0 + w_, node.y0 + h_, w_, h_, node.max_size, m4)
    recursive_subdivide(n4, k)

    node.children = [n1, n2, n3, n4]
    # clean up
    node.isLeaf = False
    node.model = None

def find_children(node):
   if node.isLeaf:
       return [node]
   else:
       children = []
       for child in node.children:
           children += (find_children(child))
   return children
