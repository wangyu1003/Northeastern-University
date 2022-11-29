import Environment
import numpy as np

env = Environment.Env('Ebone-Evaluate')


class ecmp(object):
    def __init__(self):
        self.pair_idx_to_sd = env.pair_idx_to_sd
        self.pair_sd_to_idx = env.pair_sd_to_idx
        self.num_nodes = env.num_nodes
        self.num_links = env.num_links
        self.link_idx_to_sd = env.link_idx_to_sd
        self.link_sd_to_idx = env.link_sd_to_idx
        self.link_capacities = env.link_capacities
        self.link_weights = env.link_weights
        self.shortest_paths_node = env.shortest_paths_node
        self.graph_state = env.graph_state
        self.get_ecmp_next_hops()
        self.flag = False

    def get_ecmp_next_hops(self):
        self.ecmp_next_hops = {}
        for src in range(self.num_nodes):
            for dst in range(self.num_nodes):
                if src == dst:
                    continue
                self.ecmp_next_hops[src, dst] = []
                for p in self.shortest_paths_node[self.pair_sd_to_idx[(src, dst)]]:
                    if p[1] not in self.ecmp_next_hops[src, dst]:
                        self.ecmp_next_hops[src, dst].append(p[1])

    def ecmp_next_hop_distribution(self, link_loads, demand, source, destination):
        if source == destination:
            return

        ecmp_next_hops = self.ecmp_next_hops[source, destination]

        next_hops_cnt = len(ecmp_next_hops)
        ecmp_demand = demand / next_hops_cnt
        for np in ecmp_next_hops:
            if np == destination:
                if source < np:
                    link_loads[self.link_sd_to_idx[(source, np)]] += ecmp_demand
                if source > np:
                    link_loads[self.link_sd_to_idx[(np, source)]] += ecmp_demand
                break
            if np > source:
                link_loads[self.link_sd_to_idx[(source, np)]] += ecmp_demand
            if np < source:
                link_loads[self.link_sd_to_idx[(np, source)]] += ecmp_demand

            self.ecmp_next_hop_distribution(link_loads, ecmp_demand, np, destination)

    def ecmp_traffic_distribution(self, source, destination, demand):
        link_loads = np.zeros(self.num_links)
        self.ecmp_next_hop_distribution(link_loads, demand, source, destination)
        return link_loads

    def eval_ecmp(self, source, destination, demand, link_capacities, link_loads):
        eval_link_loads = self.ecmp_traffic_distribution(source, destination, demand)
        link_capacities = link_capacities - eval_link_loads
        link_loads += eval_link_loads

        max_utilization = np.max(link_loads / (link_capacities + link_loads))
        delay = sum(link_loads / link_capacities)
        return max_utilization, delay, link_capacities, link_loads

    def robust_ecmp(self, source, destination, demand, link_capacities, link_loads, link_memory, time_interval):
        self.flag = False
        eval_link_loads = self.ecmp_traffic_distribution_robust(source, destination, demand, link_memory, time_interval)
        link_capacities = link_capacities - eval_link_loads
        link_loads += eval_link_loads
        return link_capacities, link_loads, self.flag

    def ecmp_traffic_distribution_robust(self, source, destination, demand, link_memory, time_interval):
        link_loads = np.zeros(self.num_links)
        self.ecmp_next_hop_distribution_robust(link_loads, demand, source, destination, link_memory, time_interval)
        return link_loads

    def ecmp_next_hop_distribution_robust(self, link_loads, demand, source, destination, link_memory, time_interval):
        if source == destination:
            return
        ecmp_next_hops = self.ecmp_next_hops[source, destination]
        next_hops_cnt = len(ecmp_next_hops)
        ecmp_demand = demand / next_hops_cnt
        for np in ecmp_next_hops:
            if np == destination:
                if source < np:
                    link_loads[self.link_sd_to_idx[(source, np)]] += ecmp_demand
                    k = 0
                    while k < len(link_memory):
                        if time_interval[k] > 0:
                            if link_memory[k] == self.link_sd_to_idx[(source, np)]:
                                self.flag = True
                                break
                        k += 1
                if source > np:
                    link_loads[self.link_sd_to_idx[(np, source)]] += ecmp_demand
                    for link in link_memory:
                        if link == self.link_sd_to_idx[(np, source)]:
                            self.flag = True
                            break
                break
            if np > source:
                link_loads[self.link_sd_to_idx[(source, np)]] += ecmp_demand
                for link in link_memory:
                    if link == self.link_sd_to_idx[(source, np)]:
                        self.flag = True
                        break
            if np < source:
                link_loads[self.link_sd_to_idx[(np, source)]] += ecmp_demand
                for link in link_memory:
                    if link == self.link_sd_to_idx[(np, source)]:
                        self.flag = True
                        break
            self.ecmp_next_hop_distribution_robust(link_loads, ecmp_demand, np, destination, link_memory, time_interval)
