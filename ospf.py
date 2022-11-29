from Environment import Env
from Environment import Topology
import numpy as np
import networkx as nx

env = Env('Ebone-Evaluate')
topo = Topology('Ebone-Evaluate')


class ospf(object):
    def __init__(self, config):
        self.G = topo.G
        self.data_dir = './data/'
        self.topology_file = self.data_dir + config
        self.num_nodes = env.num_nodes
        self.num_links = env.num_links
        self.link_capacities = env.link_capacities
        self.link_weights = env.link_weights
        self.link_idx_to_sd = env.link_idx_to_sd
        self.link_sd_to_idx = env.link_sd_to_idx

    def ospf_operation(self, source, destination, demand, link_capacities):
        weight = np.zeros(self.num_links)

        for i in range(self.num_links):
            if link_capacities[i] <= demand:
                weight[i] = 100
            else:
                weight[i] = self.link_weights[i]
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
                self.link_weights[j] = weight[j]
                self.G.add_weighted_edges_from([(int(s), int(d), weight[j])])
                j = j + 1

        shortest_path = nx.shortest_path(self.G, source, destination, weight='weight')
        i = 0
        j = 1
        eval_link_loads = np.zeros(self.num_links)

        while j < len(shortest_path):
            if shortest_path[i] < shortest_path[j]:
                eval_link_loads[self.link_sd_to_idx[shortest_path[i], shortest_path[j]]] += demand
                link_capacities[self.link_sd_to_idx[shortest_path[i], shortest_path[j]]] -= demand
            if shortest_path[i] > shortest_path[j]:
                eval_link_loads[self.link_sd_to_idx[shortest_path[j], shortest_path[i]]] += demand
                link_capacities[self.link_sd_to_idx[shortest_path[j], shortest_path[i]]] -= demand
            i = i + 1
            j = j + 1
        return eval_link_loads

    def eval_ospf(self, source, destination, demand, link_capacities, link_loads, eval_max_utilization, eval_delay):

        eval_link_loads = self.ospf_operation(source, destination, demand, link_capacities)
        link_capacities = link_capacities - eval_link_loads
        link_loads += eval_link_loads

        for i in range(len(link_capacities)):
            if link_capacities[i] <= 0 :
                c = link_capacities[:] + eval_link_loads[:]
                l = link_loads[:] - eval_link_loads[:]
                m = eval_max_utilization
                d = eval_delay
                return m, d, c, l

        max_utilization = np.max(link_loads / (link_capacities + link_loads))

        delay = sum(link_loads / link_capacities)

        return max_utilization, delay, link_capacities, link_loads

    def robust_ospf(self, source, destination, demand, link_capacities, link_loads):
        eval_link_loads = self.ospf_operation1(source, destination, demand, link_capacities)
        link_capacities = link_capacities - eval_link_loads

        link_loads += eval_link_loads

        return link_capacities, link_loads

    def ospf_operation1(self, source, destination, demand, link_capacities):
        shortest_path = nx.shortest_path(self.G, source, destination, weight='weight')
        i = 0
        j = 1
        eval_link_loads = np.zeros(self.num_links)
        while j < len(shortest_path):
            if shortest_path[i] < shortest_path[j]:
                eval_link_loads[self.link_sd_to_idx[shortest_path[i], shortest_path[j]]] += demand
                link_capacities[self.link_sd_to_idx[shortest_path[i], shortest_path[j]]] -= demand
            if shortest_path[i] > shortest_path[j]:
                eval_link_loads[self.link_sd_to_idx[shortest_path[j], shortest_path[i]]] += demand
                link_capacities[self.link_sd_to_idx[shortest_path[j], shortest_path[i]]] -= demand
            i = i + 1
            j = j + 1
        return eval_link_loads
