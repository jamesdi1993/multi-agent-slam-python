from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import os

batch_size = 100

class GroundTruthReader():

    def __init__(self, gt_path, cov_path):
        self.gt = np.loadtxt(gt_path, delimiter = ",")
        self.cov = np.loadtxt(cov_path, delimiter = ",")
        self.index = batch_size

        print("Ground truth shape: %s" % (self.gt.shape,))
        print("Cov shape: %s" % (self.cov.shape,))


    def step(self):

        t = self.gt[:self.index, 0]
        xs = self.gt[:self.index, 1]
        ys = self.gt[:self.index, 2]
        zs = self.gt[:self.index, 3]

        self.index += batch_size

        # print("ys: %s; xs: %s" % (ys.shape, xs.shape))

        return t, xs, ys

    def size(self):
        return self.gt.shape[0]

class SequenceReader():
    def __init__(self, data_path, data_format="csv"):
        self.data_format = data_format
        if self.data_format == "csv":
            self.data = np.loadtxt(data_path, delimiter=",")
        elif self.data_format == "npz":
            self.data = np.load(data_path)["arr_0"]
        self.index = batch_size

        print("Ground truth shape: %s" % (self.data.shape,))

    def step(self):
        t = self.data[:self.index, 0]
        xs = self.data[:self.index, 1082]
        ys = self.data[:self.index, 1083]

        self.index += batch_size
        # print("ys: %s; xs: %s" % (ys.shape, xs.shape))
        return t, xs, ys

    def size(self):
        return self.data.shape[0]

class TrajectoryReader():

    def __init__(self, data_path):
        self.data = np.load(data_path)
        self.index = 0

    def step(self):
        xs = self.data[:self.index, 0]
        ys = self.data[:self.index, 1]

        self.index += batch_size
        # print("ys: %s; xs: %s" % (ys.shape, xs.shape))
        return xs, ys

    def size(self):
        return self.data.shape[0]

# dates = ["2012-01-08", "2012-04-29"]
# dates = ["2012-01-08", "2012-04-29", "2012-06-15", "2012-08-04", "2013-04-05"]
dates = ["2012-01-08", "2012-04-29", "2012-06-15", "2012-08-04", "2013-04-05",
         "2012-01-22", "2012-02-04", "2012-03-17", "2012-03-25", "2012-09-28"]

n_agents = len(dates)

"""
# For reading ground-truth data.
data_dir = "/home/jamesdi1993/datasets/NCLT"
gt_path_format = data_dir + "/{0}/groundtruth_{0}.csv"
cov_path_format = data_dir + "/{0}/cov_{0}.csv"

gt_paths = []
cov_paths = []

for date in dates:
    gt_path = gt_path_format.format(date)
    cov_path = cov_path_format.format(date)
    gt_paths.append(gt_path)
    cov_paths.append(cov_path)
"""


"""
# For reading sequenced data
"""
data_dir = "/home/jamesdi1993/datasets/NCLT"
seq_num = 6
seq_path_format = data_dir + "/sequences/{0}_seq_{1}.npz"
seq_paths = []
for date in dates:
    seq_path = seq_path_format.format(date, seq_num)
    seq_paths.append(seq_path)
    print(seq_path)

#------------------------------------------------------------
# set up figure and animation
fig = plt.figure(figsize=(10, 15))
# fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax = fig.add_subplot(111, aspect='equal', autoscale_on=True)

ax.set_xlim(-800, 150)
ax.set_ylim(-600, 200)

ax.set_xlabel('East (m)')
ax.set_ylabel('North (m)')

# title = ax.text(0.5,0.85, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
#                 transform=ax.transAxes, ha="center")

title = ax.set_title("Ro: 0")

time_text = ""

# waypoints holds the locations of the points

waypoints = []
# gt_readers = []
sequence_readers = []

T = 0

for i, date in enumerate(dates):
    waypoints_i, = ax.plot([], [], 'o', ms=1, label='agent %d' % i)
    waypoints.append(waypoints_i)


    # gt_reader = GroundTruthReader(gt_paths[i], cov_paths[i])
    # gt_readers.append(gt_reader)
    #
    # T = max(gt_reader.size(), T)

    seq_reader = SequenceReader(seq_paths[i], data_format="npz")
    sequence_readers.append(seq_reader)
    T = max(seq_reader.size(), T)

print("Total number of frame: %s" % (T/batch_size))

def init():
    """initialize animation"""
    global waypoints

    for waypoints_i in waypoints:
        waypoints_i.set_data([], [])
    return waypoints

def animate(i):
    """perform animation step"""
    global sequence_readers, waypoints, time_text

    ts = []

    for j, waypoints_j in enumerate(waypoints):
        # gt_reader = gt_readers[j]
        # t, xs, ys = gt_reader.step()

        seq_reader = sequence_readers[j]
        t, xs, ys = seq_reader.step()

        # update pieces of the animation
        waypoints_j.set_data(-ys, xs)

        ts.append(t[-1])

    text_arr = ["Agent %d: %s" % (i, ts[i]) for i in range(len(dates))]

    if i % 10 == 0:
        time_text = time_text + ("Agent's timestamp at: %d\n" % i) + "; ".join(text_arr) + "\n"
        # print(time_text)
        print("Agent's timestamp at: %d" % i)
        print(text_arr)
    title.set_text("Timestamp: %s" % i)
    return waypoints

ani = FuncAnimation(fig, animate, frames=T // batch_size, interval=100, init_func=init, repeat=False)
plt.legend(markerscale=6)
output_path = os.path.join(data_dir, 'trajectory_{0}_seq{1}.mp4'.format(n_agents, seq_num))

ani.save(output_path, fps=10, extra_args=['-vcodec', 'libx264'])

plt.show()
timestamp_path =  os.path.join(data_dir, 'time_{}.txt'.format(n_agents))
print("Writing to %s" % timestamp_path)
with open(timestamp_path, "w") as text_file:
    text_file.write(time_text)


# ani.save(output_path, fps=10)


