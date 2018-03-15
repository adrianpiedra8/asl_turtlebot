from astar import AStar
from astar import DetOccupancyGrid2D
import numpy as np
import matplotlib.pyplot as plt
import itertools
import pdb

def find_closest(x_init, x_goals, width, height, occupancy):
    min_dist = np.float('inf')
    closest = None
    for x_goal in x_goals:
        astar = AStar((0, 0), (width, height), x_init, x_goal, occupancy)
 
        if astar.solve() and len(astar.path) < min_dist:
            min_dist = len(astar.path)
            closest = x_goal

    return closest, min_dist

def traveling_salesman_fast(x_init, x_goals, width, height, occupancy):
    wps = [x_init]
    circuit_length = 0
    for i in range(len(x_goals)):
        closest, curr_length = find_closest(x_init, x_goals, width, height, occupancy)
        circuit_length += curr_length

        if closest == None:
            print "Could not find circuit"
            return wps

        x_init = closest
        wps.append(x_init)
        x_goals.remove(closest)

    print(circuit_length)
    return wps


def get_circuit_length(circuit, paths):
    circuit_length = 0 
    for i in range(len(circuit) - 1):
        node1 = circuit[i]
        node2 = circuit[i+1]
        pair = (node1, node2)
        if node1 > node2:
            pair = (node2, node1)

        try:
            circuit_length += paths[pair]
        except:
            pdb.set_trace()

    return circuit_length

def traveling_salesman_exact(x_init, x_goals, width, height, occupancy):

    paths = {}
    for i in range(len(x_goals)):
        astar = AStar((0, 0), (width, height), x_init, x_goals[i], occupancy)
        if astar.solve():
            paths[(0, i+1)] = len(astar.path)
        else:
            paths[(0,i+1)] = np.float('inf')

    for pair in itertools.combinations(range(len(x_goals)), 2):
        astar = AStar((0, 0), (width, height), x_goals[pair[0]], x_goals[pair[1]], occupancy)
        pair = [x+1 for x in list(pair)]
        pair = tuple(pair)
        if astar.solve():
            paths[pair] = len(astar.path)
        else:
            paths[pair] = np.float('inf')

    min_length = np.float('inf')
    for circuit in itertools.permutations(range(len(x_goals))):

        circuit = [x+1 for x in list(circuit)]
        circuit = [0] + circuit
        curr_length = get_circuit_length(circuit, paths) 
        if curr_length < min_length:
            min_length = curr_length
            min_circuit = circuit

    wps = [x_init] + x_goals

    wps = [wps[i] for i in min_circuit]

    print(min_length)
    return wps
