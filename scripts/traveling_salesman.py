from astar import AStar
from astar import DetOccupancyGrid2D
import numpy as np
import matplotlib.pyplot as plt
import itertools
import pdb

def find_closest(x_init, x_goals, statespace_lo, statespace_hi, occupancy, resolution):
    min_dist = np.float('inf')
    closest = None
    for x_goal in x_goals:
        astar = AStar(statespace_lo, statespace_hi, x_init, x_goal, occupancy, resolution)
 
        if astar.solve() and len(astar.path) < min_dist:
            min_dist = len(astar.path)
            closest = x_goal

    return closest, min_dist

def traveling_salesman_fast(x_init, x_orig_goals, statespace_lo, statespace_hi, occupancy, resolution):
    wps = []
    circuit = []
    x_goals = list(x_orig_goals) # make a copy
    circuit_length = 0
    for i in range(len(x_goals)):
        closest, curr_length = find_closest(x_init, x_goals, statespace_lo, statespace_hi, occupancy, resolution)
        circuit_length += curr_length

        if closest == None:
            print "Could not find circuit"
            circuit = [x_orig_goals.index(x) for x in wps]
            return circuit

        x_init = closest
        wps.append(x_init)
        x_goals.remove(closest)

    circuit = [x_orig_goals.index(x) for x in wps]
    print("Found path with length: {}:".format(circuit_length))
    print(circuit)
    print(wps)
    return circuit

def get_circuit_length(circuit, paths):
    circuit_length = 0 
    for i in range(len(circuit) - 1):
        node1 = circuit[i]
        node2 = circuit[i+1]
        pair = (node1, node2)
        if node1 > node2:
            pair = (node2, node1)

        circuit_length += paths[pair]


    return circuit_length

def traveling_salesman_exact(x_init, x_goals, statespace_lo, statespace_hi, occupancy, resolution):
    paths = {}
    for i in range(len(x_goals)):
        astar = AStar(statespace_lo, statespace_hi, x_init, x_goals[i], occupancy, resolution)
        if astar.solve():
            paths[(0, i+1)] = len(astar.path)
        else:
            paths[(0,i+1)] = np.float('inf')

    for pair in itertools.combinations(range(len(x_goals)), 2):
        astar = AStar(statespace_lo, statespace_hi, x_goals[pair[0]], x_goals[pair[1]], occupancy, resolution)
        pair = [x+1 for x in list(pair)]
        pair = tuple(pair)
        if astar.solve():
            paths[pair] = len(astar.path)
        else:
            paths[pair] = np.float('inf')

    min_length = np.float('inf')
    min_circuit = [0]
    for circuit in itertools.permutations(range(len(x_goals))):

        circuit = [x+1 for x in list(circuit)]
        circuit = [0] + circuit
        curr_length = get_circuit_length(circuit, paths) 
        if curr_length < min_length:
            min_length = curr_length
            min_circuit = circuit



    min_circuit = min_circuit[1:]
    min_circuit = [x-1 for x in min_circuit]

    wps = [x_goals[i] for i in min_circuit]

    print("Found path with length: {}:".format(min_length))
    print(min_circuit)
    print(wps)

    return min_circuit
