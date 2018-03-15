from astar import AStar
from astar import DetOccupancyGrid2D
import numpy as np
import matplotlib.pyplot as plt
import traveling_salesman
import pdb

# A large random example
width = 51
height = 51
num_obs = 8
min_size = 15
max_size = 25
obs_corners_x = np.random.randint(0,width,num_obs)
obs_corners_y = np.random.randint(0,height,num_obs)
obs_lower_corners = np.vstack([obs_corners_x,obs_corners_y]).T
obs_sizes = np.random.randint(min_size,max_size,(num_obs,2))
obs_upper_corners = obs_lower_corners + obs_sizes
obstacles = zip(obs_lower_corners,obs_upper_corners)
occupancy = DetOccupancyGrid2D(width, height, obstacles)
x_goals = []
for i in range(8):
    x_goal = tuple(np.random.randint(0,height-2,2).tolist())
    while not (occupancy.is_free(x_goal)):
        x_goal = tuple(np.random.randint(0,height-2,2).tolist())
    x_goals.append(x_goal)

ts_fast = traveling_salesman.traveling_salesman_fast(x_goals[0], x_goals[1:], width, height, occupancy)

fig1 = plt.figure()
for i in range(len(ts_fast) - 1):
    astar = AStar((0, 0), (width, height), ts_fast[i], ts_fast[i+1], occupancy)
    if astar.solve():
        astar.plot_path(fig1, r"$x_{"+str(i)+"}$", r"$x_{"+str(i+1)+"}$" + str(len(astar.path)), pcolor='blue')

fig2 = plt.figure()
ts_exact = traveling_salesman.traveling_salesman_exact(x_goals[0], x_goals[1:], width, height, occupancy)
for i in range(len(ts_exact) - 1):
    astar = AStar((0, 0), (width, height), ts_exact[i], ts_exact[i+1], occupancy)
    if astar.solve():
        astar.plot_path(fig2, r"$x_{"+str(i)+"}$", r"$x_{"+str(i+1)+"}$" + str(len(astar.path)), pcolor='red')

plt.show()