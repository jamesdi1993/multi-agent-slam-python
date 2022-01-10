from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import os

batch_size = 100

class TrajectoryReader():

    def __init__(self, data_path):
        self.data = np.load(data_path)['arr_0']
        self.index = 0

    def step(self):
        ts = self.data[:, :self.index, 0]
        xs = self.data[:, :self.index, 1]
        ys = self.data[:, :self.index, 2]

        self.index += batch_size
        return ts, xs, ys

    def size(self):
        return self.data.shape[1]

dates = ["2012-01-08", "2012-04-29", "2012-06-15", "2012-08-04", "2013-04-05",
         "2012-01-22", "2012-02-04", "2012-03-17", "2012-03-25", "2012-09-28"]

n_agents = len(dates)
time_text = ""

"""
# For reading sequenced data
"""
data_dir = "/home/jamesdi1993/datasets/NCLT"
data_path = data_dir + "/sequences/trajectories_10.npz"
reader = TrajectoryReader(data_path)

#------------------------------------------------------------
# set up figure and animation
fig = plt.figure(figsize=(10, 15))
# fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax = fig.add_subplot(111, aspect='equal', autoscale_on=True)

ax.set_xlim(-800, 150)
ax.set_ylim(-600, 200)

ax.set_xlabel('East (m)')
ax.set_ylabel('North (m)')
title = ax.set_title("Ro: 0")

# waypoints holds the locations of the points

waypoints = []
sequence_readers = []

T = reader.size()

for i, date in enumerate(dates):
    waypoints_i, = ax.plot([], [], 'o', ms=1, label='agent %d' % i)
    waypoints.append(waypoints_i)

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
    t, xs, ys = reader.step()
    for j, waypoints_j in enumerate(waypoints):
        # gt_reader = gt_readers[j]
        # t, xs, ys = gt_reader.step()

        # update pieces of the animation
        waypoints_j.set_data(-ys[j, :], xs[j, :])
        ts.append(t)

    text_arr = ["Agent %d: %s" % (i, ts[i]) for i in range(len(dates))]

    if i % 10 == 0:
        time_text = time_text + ("Agent's timestamp at: %d\n" % i) + "; ".join(text_arr) + "\n"
        print(time_text)
        print("Agent's timestamp at: %d" % i)
        # print(text_arr)
    title.set_text("Timestamp: %s" % i)
    return waypoints

ani = FuncAnimation(fig, animate, frames=T // batch_size, interval=100, init_func=init, repeat=False)
plt.legend(markerscale=6)
output_path = os.path.join(data_dir, 'trajectory_{}.mp4'.format(n_agents))

plt.show()
timestamp_path =  os.path.join(data_dir, 'time_{}.txt'.format(n_agents))


