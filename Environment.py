from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gc
import os

import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt


def create_Ebone_graph():
    Gbase = nx.Graph()
    Gbase.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22])
    Gbase.add_edges_from(
        [(0, 1), (0, 2), (0, 3), (0, 4), (0, 6), (1, 4), (2, 10), (2, 13), (2, 14), (2, 20), (4, 13), (4, 14),
         (4, 15), (4, 16), (5, 6), (5, 7), (6, 10), (7, 11), (8, 9), (8, 10), (8, 11), (8, 12),
         (8, 14), (9, 10), (9, 12), (10, 11), (10, 14), (10, 15), (11, 12), (11, 19), (12, 19),
         (13, 16), (14, 15), (15, 18), (16, 20), (17, 18), (18, 22), (19, 21)])
    return Gbase


def create_Sprintlink_graph():
    Gbase = nx.Graph()
    Gbase.add_nodes_from(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
         27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43])
    Gbase.add_edges_from(
        [(0, 1), (0, 2), (0, 3), (0, 4), (0, 8), (0, 43), (0, 18), (1, 6), (1, 8), (1, 18), (1, 17), (1, 30), (2, 10),
         (2, 20),
         (2, 21), (2, 32), (2, 33), (3, 5), (3, 6), (3, 9), (3, 19), (3, 26), (3, 14), (3, 37), (3, 38), (3, 8),
         (4, 10),
         (4, 18), (4, 28), (4, 32), (4, 6), (4, 8), (4, 19), (4, 20), (4, 34), (5, 6), (5, 7), (5, 11), (6, 8), (6, 11),
         (6, 26), (6, 28), (6, 31), (6, 19), (6, 34), (7, 22), (7, 40), (7, 11), (8, 19), (8, 20), (8, 39), (8, 26),
         (8, 28), (8, 31), (10, 26), (11, 12), (11, 13), (11, 16), (12, 23), (14, 19), (15, 16), (15, 36),
         (16, 36), (17, 18), (18, 21), (18, 33), (18, 29), (18, 26), (18, 35), (19, 20), (19, 31), (20, 25), (20, 28),
         (21, 42), (21, 35), (22, 36), (22, 23), (22, 24), (27, 28), (28, 32), (26, 28), (26, 34), (34, 41)])
    return Gbase


def create_Tiscali_graph():
    Gbase = nx.Graph()
    Gbase.add_nodes_from(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
         27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48])
    Gbase.add_edges_from(
        [(0, 1), (0, 2), (0, 3), (4, 5), (4, 6), (4, 7), (8, 9), (10, 11), (7, 12), (7, 13), (7, 8), (7, 14), (15, 16),
         (17, 18), (17, 19), (20, 21), (5, 14), (24, 25), (26, 27), (27, 28), (8, 29), (8, 14), (8, 27), (9, 31),
         (18, 33), (18, 32),
         (18, 34), (36, 37), (14, 31), (14, 38), (18, 30), (18, 39), (18, 40), (18, 19), (18, 27), (41, 42), (18, 43),
         (6, 8), (6, 14),
         (1, 27), (1, 8), (1, 14), (1, 18), (1, 29), (1, 17), (1, 42), (1, 19), (1, 28), (1, 3), (1, 26), (8, 31),
         (8, 44), (8, 38), (18, 42), (8, 12), (11, 18), (14, 45), (44, 46), (2, 27), (2, 14), (16, 27), (11, 20),
         (11, 25), (11, 36), (11, 23), (11, 48), (19, 26), (12, 14), (14, 44), (14, 35), (14, 17), (14, 41), (14, 19),
         (14, 28), (14, 27), (14, 47), (14, 22), (14, 18), (3, 29), (3, 27), (3, 8), (0, 14), (8, 13), (3, 14), (3, 19),
         (15, 27)])
    return Gbase


def generate_nx_graph(self, config, data_dir='./data/'):
    self.topology_file = data_dir + config
    if config == 'Ebone' or config == 'Ebone-Evaluate':
        G = create_Ebone_graph()
    elif config == 'Sprintlink' or config == 'Sprintlink-Evaluate':
        G = create_Sprintlink_graph()
    elif config == 'Tiscali' or config == 'Tiscali-Evaluate':
        G = create_Tiscali_graph()
    topology_file = data_dir + config
    f = open(topology_file, 'r')
    header = f.readline()
    num_links = int(header[header.find(':', 10) + 2:])
    f.readline()
    link_capacities = np.empty((num_links))
    for line in f:
        link = line.split('\t')
        i, s, d, w, c = link
        link_capacities[int(i)] = float(c)

    linkID = 1  
    for s, d in G.edges():
        G.get_edge_data(s, d)['edgeId'] = linkID
        G.get_edge_data(s, d)["capacity"] = link_capacities[s]
        G.get_edge_data(s, d)['link_utilization'] = 0
        G.get_edge_data(s, d)['bw_allocated'] = 0
        linkID = linkID + 1

    return G


class Topology(object):
    def __init__(self, config, data_dir='./data/'):
        self.num_links = None
        self.num_nodes = None
        self.topology_file = data_dir + config
        self.shortest_paths_file = self.topology_file + '_shortest_paths'
        self.G = nx.Graph()
        self.load_topology()
        self.calculate_paths()

    def load_topology(self):
        f = open(self.topology_file, 'r')
        header = f.readline()
        self.num_nodes = int(header[header.find(':') + 2:header.find('\t')])
        numlinks = int(header[header.find(':', 10) + 2:])
        self.num_links = int(numlinks / 2)
        f.readline()
        self.link_idx_to_sd = {}
        self.link_sd_to_idx = {}
        self.link_capacities = np.empty((self.num_links))
        self.link_weights = np.empty((self.num_links))
        j = 0
        for line in f:
            link = line.split('\t')
            i, s, d, w, c = link
            if int(d) > int(s):
                self.link_idx_to_sd[j] = (int(s), int(d))
                self.link_sd_to_idx[(int(s), int(d))] = j
                self.link_capacities[j] = float(c)
                self.link_weights[j] = int(w)
                j = j + 1
                self.G.add_weighted_edges_from([(int(s), int(d), int(w))])
        f.close()
    # nx.draw_networkx(self.G)
    # plt.show()

    def calculate_paths(self):
        self.pair_idx_to_sd = []
        self.pair_sd_to_idx = {}
        self.shortest_paths = []
        if os.path.exists(self.shortest_paths_file):
            f = open(self.shortest_paths_file, 'r')
            self.num_pairs = 0
            for line in f:
                sd = line[:line.find(':')]
                s = int(sd[:sd.find('-')])
                d = int(sd[sd.find('>') + 1:])
                self.pair_idx_to_sd.append((s, d))
                self.pair_sd_to_idx[(s, d)] = self.num_pairs
                self.num_pairs += 1
                self.shortest_paths.append([])
                paths = line[line.find(':') + 1:].strip()[1:-1]
                while paths != '':
                    idx = paths.find(']')
                    path = paths[1:idx]
                    node_path = np.array(path.split(',')).astype(np.int16)
                    assert node_path.size == np.unique(node_path).size
                    self.shortest_paths[-1].append(node_path)
                    paths = paths[idx + 3:]
        else:
            f = open(self.shortest_paths_file, 'w+')
            self.num_pairs = 0
            for s in range(self.num_nodes):
                for d in range(self.num_nodes):
                    if s != d:
                        self.pair_idx_to_sd.append((s, d))
                        self.pair_sd_to_idx[(s, d)] = self.num_pairs
                        self.num_pairs += 1
                        self.shortest_paths.append(list(nx.all_shortest_paths(self.G, s, d, weight='weight')))
                        line = str(s) + '->' + str(d) + ': ' + str(self.shortest_paths[-1])
                        f.writelines(line + '\n')

        assert self.num_pairs == self.num_nodes * (self.num_nodes - 1)
        f.close()


class Env(object):
    def __init__(self, config):
        self.data_dir = './data/'
        self.topology = Topology(config, self.data_dir)
        self.num_pairs = self.topology.num_pairs
        self.pair_idx_to_sd = self.topology.pair_idx_to_sd
        self.pair_sd_to_idx = self.topology.pair_sd_to_idx
        self.num_nodes = self.topology.num_nodes
        self.num_links = self.topology.num_links
        self.link_idx_to_sd = self.topology.link_idx_to_sd
        self.link_sd_to_idx = self.topology.link_sd_to_idx
        self.link_capacities = self.topology.link_capacities
        self.link_weights = self.topology.link_weights
        self.shortest_paths_node = self.topology.shortest_paths

        self.first = None
        self.firstTrueSize = None
        self.second = None

        self.graph = None
        self.initial_state = None
        self.link_feature = None
        self.K = 3
        self.source = None
        self.destination = None
        self.nodes = None
        self.demand = []

        self.listofDemands = None
        self.graph_state = None
        self.diameter = None
        self.ordered_edges = None
        self.edgesDict = None
        self.state = None
        self.step_over = True
        self.reward = 0
        self.topology_file = self.data_dir + config
        self.shortest_paths_file = self.topology_file + '_shortest_paths'
        self.G = nx.Graph()
        self.load_topology()
        self.allPaths = dict()
        self.shortest_paths = []
        self.pathList = []
        self.p = 0

    def load_topology(self):
        f = open(self.topology_file, 'r')
        header = f.readline()
        self.num_nodes = int(header[header.find(':') + 2:header.find('\t')])
        numlinks = int(header[header.find(':', 10) + 2:])
        self.num_links = int(numlinks / 2)
        f.readline()
        self.link_idx_to_sd = {}
        self.link_sd_to_idx = {}
        self.link_capacities = np.empty((self.num_links))
        self.link_weights = np.empty((self.num_links))
        j = 0
        for line in f:
            link = line.split('\t')
            i, s, d, w, c = link
            if int(d) > int(s):
                self.link_idx_to_sd[j] = (int(s), int(d))
                self.link_sd_to_idx[(int(s), int(d))] = j
                self.link_capacities[j] = float(c)
                self.link_weights[j] = int(w)
                j = j + 1
                self.G.add_weighted_edges_from([(int(s), int(d), int(w))])

        f.close()

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def num_shortest_path(self, topology):
        self.diameter = nx.diameter(self.graph)
        # Iterate over all node1,node2 pairs from the graph
        for n1 in range(self.num_nodes):
            for n2 in range(self.num_nodes):
                if (n1 != n2):
                    if str(n1) + ':' + str(n2) not in self.allPaths:
                        self.allPaths[str(n1) + ':' + str(n2)] = []

                    [self.allPaths[str(n1) + ':' + str(n2)].append(p) for p in
                     (nx.all_shortest_paths(self.G, n1, n2, weight='weight'))]

                    path = 0
                    while path < self.K and path < len(self.allPaths[str(n1) + ':' + str(n2)]):
                        currentPath = self.allPaths[str(n1) + ':' + str(n2)][path]
                        path = path + 1
                    gc.collect()

    def _first_second_between(self):
        self.first = list()
        self.second = list()

        # For each edge we iterate over all neighbour edges
        for i, j in self.ordered_edges:
            neighbour_edges = self.graph.edges(i)

            for m, n in neighbour_edges:
                if ((i != m or j != n) and (i != n or j != m)):
                    self.first.append(self.edgesDict[str(i) + ':' + str(j)])
                    self.second.append(self.edgesDict[str(m) + ':' + str(n)])

            neighbour_edges = self.graph.edges(j)
            for m, n in neighbour_edges:
                if ((i != m or j != n) and (i != n or j != m)):
                    self.first.append(self.edgesDict[str(i) + ':' + str(j)])
                    self.second.append(self.edgesDict[str(m) + ':' + str(n)])

    def generate_environment(self, topology, listofdemands):
        self.graph = generate_nx_graph(topology, 'Ebone-Evaluate')
        self.listofDemands = listofdemands
        self.num_shortest_path(topology)

        self.edgesDict = dict()
        some_edges_1 = [tuple(sorted(edge)) for edge in self.graph.edges()]
        self.ordered_edges = sorted(some_edges_1)
        self.numNodes = len(self.graph.nodes())
        self.numEdges = len(self.graph.edges())

        self.graph_state = np.zeros((self.num_links, 3))
        self.link_feature = np.zeros(self.num_links)
        position = 0
        for edge in self.ordered_edges:
            i = edge[0]
            j = edge[1]
            self.edgesDict[str(i) + ':' + str(j)] = position
            self.edgesDict[str(j) + ':' + str(i)] = position
            self.graph_state[position][2] = self.graph.get_edge_data(i, j)["bw_allocated"]
            self.graph_state[position][0] = self.graph.get_edge_data(i, j)["capacity"]
            self.graph.get_edge_data(i, j)['link_utilization'] = self.graph_state[position][2] / \
                                                                 self.graph_state[position][0]
            self.graph_state[position][1] = self.graph.get_edge_data(i, j)['link_utilization']

            self.link_feature[position] = self.graph.get_edge_data(i, j)['link_utilization']
            position = position + 1

        self.initial_state = np.copy(self.graph_state)

        self._first_second_between()
        self.firstTrueSize = len(self.first)

        self.nodes = list(range(0, self.num_nodes))

    def step(self, state, action, source, destination, demand):
        self.graph_state = np.copy(state)
        self.step_over = True
        self.reward = 0
        self.p = 0
        j = 0
        f = open(self.topology_file, 'r')
        header = f.readline()
        self.num_nodes = int(header[header.find(':') + 2:header.find('\t')])
        numlinks = int(header[header.find(':', 10) + 2:])
        self.num_links = int(numlinks / 2)
        f.readline()
        self.link_idx_to_sd = {}
        self.link_sd_to_idx = {}
        self.link_capacities = np.empty((self.num_links))
        self.link_weights = np.empty((self.num_links))
        for line in f:
            link = line.split('\t')
            i, s, d, w, c = link
            if int(d) > int(s):
                self.link_idx_to_sd[j] = (int(s), int(d))
                self.link_sd_to_idx[(int(s), int(d))] = j
                self.link_capacities[j] = float(c)
                self.link_weights[j] = action[0, j]
                self.G.add_weighted_edges_from([(int(s), int(d), action[0, j])])
                j = j + 1

        self.allPaths[str(source) + ':' + str(destination)] = []
        [self.allPaths[str(source) + ':' + str(destination)].append(p) for p in
         (nx.all_shortest_paths(self.G, source, destination, weight='weight'))]
        currentPath = self.allPaths[str(source) + ':' + str(destination)][0]

        i = 0
        j = 1
        while j < len(currentPath):
            self.graph_state[self.edgesDict[str(currentPath[i]) + ':' + str(currentPath[j])]][
                2] += demand
            self.graph_state[self.edgesDict[str(currentPath[i]) + ':' + str(currentPath[j])]][
                0] -= demand

            bw_l = self.graph_state[self.edgesDict[str(currentPath[i]) + ':' + str(currentPath[j])]][
                2]
            capacity_l = self.graph_state[self.edgesDict[str(currentPath[i]) + ':' + str(currentPath[j])]][
                             0] + self.graph_state[self.edgesDict[str(currentPath[i]) + ':' + str(currentPath[j])]][2]
            utilization_l = bw_l / capacity_l
            self.graph_state[self.edgesDict[str(currentPath[i]) + ':' + str(currentPath[j])]][
                1] = utilization_l
            i = i + 1
            j = j + 1
        self.reward = -np.std(self.graph_state[:, 1], ddof=1)
        self.step_over = False
        self.demand = random.choice(self.listofDemands)
        self.source = random.choice(self.nodes)
        while True:
            self.destination = random.choice(self.nodes)
            if self.destination != self.source:
                break

        return self.graph_state, self.reward, self.source, self.destination, self.demand, self.step_over

    def step_eval(self, state, action, source, destination, demand):
        self.graph_state = np.copy(state)
        self.reward = 0
        self.p = 0
        j = 0
        f = open(self.topology_file, 'r')
        header = f.readline()
        self.num_nodes = int(header[header.find(':') + 2:header.find('\t')])
        numlinks = int(header[header.find(':', 10) + 2:])
        self.num_links = int(numlinks / 2)
        f.readline()
        self.link_idx_to_sd = {}
        self.link_sd_to_idx = {}
        self.link_capacities = np.empty((self.num_links))
        self.link_weights = np.empty((self.num_links))
        for line in f:
            link = line.split('\t')
            i, s, d, w, c = link
            if int(d) > int(s):
                self.link_idx_to_sd[j] = (int(s), int(d))
                self.link_sd_to_idx[(int(s), int(d))] = j
                self.link_capacities[j] = float(c)
                self.link_weights[j] = action[0, j]
                self.G.add_weighted_edges_from([(int(s), int(d), action[0, j])])
                j = j + 1

        self.allPaths[str(source) + ':' + str(destination)] = []
    
        [self.allPaths[str(source) + ':' + str(destination)].append(p) for p in
         (nx.all_shortest_paths(self.G, source, destination, weight='weight'))]
        currentPath = self.allPaths[str(source) + ':' + str(destination)][0]
        self.pathList.append(currentPath)
        i = 0
        j = 1
        while j < len(currentPath):
            if self.graph_state[self.edgesDict[str(currentPath[i]) + ':' + str(currentPath[j])]][0] < demand:
                self.p += 1
                self.reward -= self.p
                return self.graph_state, self.reward

            self.graph_state[self.edgesDict[str(currentPath[i]) + ':' + str(currentPath[j])]][
                2] += demand
            self.graph_state[self.edgesDict[str(currentPath[i]) + ':' + str(currentPath[j])]][
                0] -= demand

            bw_l = self.graph_state[self.edgesDict[str(currentPath[i]) + ':' + str(currentPath[j])]][
                2]
            capacity_l = self.graph_state[self.edgesDict[str(currentPath[i]) + ':' + str(currentPath[j])]][
                             0] + self.graph_state[self.edgesDict[str(currentPath[i]) + ':' + str(currentPath[j])]][2]

            utilization_l = bw_l / capacity_l
            self.graph_state[self.edgesDict[str(currentPath[i]) + ':' + str(currentPath[j])]][
                1] = utilization_l
            i = i + 1
            j = j + 1

        self.reward = -np.std(self.graph_state[:, 1], ddof=1)

        return self.graph_state, self.reward

    def step_robust(self, state, action, source, destination, demand):
        self.graph_state = np.copy(state)
        self.step_over = True
        self.reward = 0
        self.p = 0
        
        j = 0
        f = open(self.topology_file, 'r')
        header = f.readline()
        self.num_nodes = int(header[header.find(':') + 2:header.find('\t')])
        numlinks = int(header[header.find(':', 10) + 2:])
        self.num_links = int(numlinks / 2)
        f.readline()
        self.link_idx_to_sd = {}
        self.link_sd_to_idx = {}
        self.link_capacities = np.empty((self.num_links))
        self.link_weights = np.empty((self.num_links))
        for line in f:
            link = line.split('\t')
            i, s, d, w, c = link
            if int(d) > int(s):
                self.link_idx_to_sd[j] = (int(s), int(d))
                self.link_sd_to_idx[(int(s), int(d))] = j
                self.link_capacities[j] = float(c)
                self.link_weights[j] = action[0, j]
                self.G.add_weighted_edges_from([(int(s), int(d), action[0, j])])
                j = j + 1

        self.allPaths[str(source) + ':' + str(destination)] = []
        [self.allPaths[str(source) + ':' + str(destination)].append(p) for p in
         (nx.all_shortest_paths(self.G, source, destination, weight='weight'))]

        currentPath = self.allPaths[str(source) + ':' + str(destination)][0]
        length = len(self.allPaths[str(source) + ':' + str(destination)])
        i = 0
        j = 1
        index = 1
        while j < len(currentPath):
            if self.graph_state[self.edgesDict[str(currentPath[i]) + ':' + str(currentPath[j])]][0] <= demand and index < length:
                self.p += 1
                self.reward -= self.p
                currentPath = self.allPaths[str(source) + ':' + str(destination)][index]
                index += 1
                i = 0
                j = 1
                continue
            self.graph_state[self.edgesDict[str(currentPath[i]) + ':' + str(currentPath[j])]][
                2] += demand
            self.graph_state[self.edgesDict[str(currentPath[i]) + ':' + str(currentPath[j])]][
                0] -= demand
            bw_l = self.graph_state[self.edgesDict[str(currentPath[i]) + ':' + str(currentPath[j])]][
                2]
            capacity_l = self.graph_state[self.edgesDict[str(currentPath[i]) + ':' + str(currentPath[j])]][
                             0] + self.graph_state[self.edgesDict[str(currentPath[i]) + ':' + str(currentPath[j])]][2]
            if capacity_l <= 0:
                utilization_l = 1
            else:
                utilization_l = bw_l / capacity_l
            self.graph_state[self.edgesDict[str(currentPath[i]) + ':' + str(currentPath[j])]][
                1] = utilization_l
            i = i + 1
            j = j + 1
        self.reward = -np.std(self.graph_state[:, 1], ddof=1)
        self.step_over = False
        self.demand = demand
        return self.graph_state, self.reward

    def reset(self):
        self.graph_state = np.copy(self.initial_state)
        self.demand = random.choice(self.listofDemands)
        self.source = random.choice(self.nodes)
        # We pick a pair of SOURCE,DESTINATION different nodes
        while True:
            self.destination = random.choice(self.nodes)
            if self.destination != self.source:
                break

        return self.graph_state, self.demand, self.source, self.destination

    def eval_sap_reset(self, demand, source, destination):
        self.graph_state = np.copy(self.initial_state)
        self.demand = demand
        self.source = source
        self.destination = destination

        return self.graph_state
