import argparse
import csv
import gc
import math
from collections import deque
from ecmp import ecmp
from ospf import ospf
import os
import random
import numpy as np
import tensorflow as tf
from Actor import Actor
from Critic import Critic
from OU import OUActionNoise
from ReplayBuffer import ReplayBuffer
from DDPG import DRLactor, Replaybuffer
from DDPG import DRLcritic
from Environment import Env
from scipy import stats
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

env = Env('Ebone-Evaluate')

SEED = 20

NUMBER_EPISODES = 7
NUMBER_SESSIONS = 20

hparams = {
    'l2': 0.1,
    'dropout_rate': 0.01,
    'link_state_dim': 20,
    'readout_units': 38,
    'LR_A': 0.000001,  # learning rate for mpnn-actor
    'batch_size': 16,
    'T': 4
}
LR_A = 0.000001  # learning rate for DDPGActor
LR_C = 0.00001  # learning rate for critic
GAMMA = 0.99  # discount rate
TAU = 0.01  # soft update rate
MEMORY_CAPACITY = 10000
MEMORY_CAPACITY1 = 1000
BATCH_SIZE = 16
MAX_QUEUE_SIZE = 1000

listofDemands = [10, 25, 40]

os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(1)


def cummin(alist, extractor):
    with tf.name_scope('cummin'):
        mines = [tf.reduce_min(extractor(v)) + 1 for v in alist]
        cummines = [tf.zeros_like(mines[0])]
        for i in range(len(mines) - 1):
            cummines.append(tf.math.add_n(mines[0:i + 1]))
    return cummines

class MDAgent(object):
    def __init__(self, a_dim):
        self.pointer = 0
        self.a_dim = a_dim
        self.bw_allocated_feature = np.zeros(env.num_links)
        self.memory = ReplayBuffer(MEMORY_CAPACITY)
        self.a_memory = deque(maxlen=MAX_QUEUE_SIZE)
        self.actor = Actor(hparams)
        self.actor.build()
        self.target_actor = Actor(hparams)
        self.target_actor.build()

        self.critic = Critic()
        self.target_critic = self.critic

        self.noise = OUActionNoise(mu=np.zeros(a_dim))
        self.a_optimizer = tf.keras.optimizers.SGD(learning_rate=hparams['LR_A'], momentum=0.9, nesterov=True)
        self.c_optimizer = tf.keras.optimizers.RMSprop(LR_C)
        self.ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)

        self.capacity_feature = None
        self.bw_allocated_feature = None

    def choose_action(self, env, state, demand, source, destination):
        listGraphs = []
        list_k_features = list()

        pathList = env.allPaths[str(source) + ':' + str(destination)]
        path = np.random.randint(0, len(pathList))
        while path < len(pathList):
            state_copy = np.copy(state)
            currentPath = pathList[path]
            i = 0
            j = 1
            while (j < len(currentPath)):
                state_copy[env.edgesDict[str(currentPath[i]) + ':' + str(currentPath[j])]][1] = demand
                i = i + 1
                j = j + 1
            listGraphs.append(state_copy)
            features = self.get_graph_features(env, state_copy)
            list_k_features.append(features)

            path = path + 1

        vs = [v for v in list_k_features]

        graph_ids = [tf.fill([tf.shape(vs[it]['link_state'])[0]], it) for it in range(len(list_k_features))]
        first_offset = cummin(vs, lambda v: v['first'])
        second_offset = cummin(vs, lambda v: v['second'])

        tensors = ({
            'graph_id': tf.concat([v for v in graph_ids], axis=0),
            'link_state': tf.concat([v['link_state'] for v in vs], axis=0),
            'first': tf.concat([v['first'] + m for v, m in zip(vs, first_offset)], axis=0),
            'second': tf.concat([v['second'] + m for v, m in zip(vs, second_offset)], axis=0),
            'num_edges': tf.math.add_n([v['num_edges'] for v in vs]),
        })

        a_predict = self.actor(tensors['link_state'], tensors['graph_id'], tensors['first'],
                                                tensors['second'], tensors['num_edges'], training=False).numpy()

        noise = self.noise()
        action_ = a_predict + np.clip(noise, 0, 10)

        return action_, list_k_features[0]

    def get_graph_features(self, env, copyGraph):
        cmax = np.max(copyGraph[:, 0])
        self.capacity_feature = copyGraph[:, 0]/cmax
        bmax = np.max(listofDemands)
        self.bw_allocated_feature = (copyGraph[:, 2]) / bmax

        sample = {
            'num_edges': env.numEdges,
            'length': env.firstTrueSize,
            'capacity': tf.convert_to_tensor(value=self.capacity_feature, dtype=tf.float32),
            'link_utilization': tf.convert_to_tensor(value=env.link_feature, dtype=tf.float32),
            'bw_allocated': tf.convert_to_tensor(value=self.bw_allocated_feature, dtype=tf.float32),
            'first': tf.convert_to_tensor(env.first, dtype=tf.int32),
            'second': tf.convert_to_tensor(env.second, dtype=tf.int32)
        }

        sample['capacity'] = tf.reshape(sample['capacity'][0:sample['num_edges']], [sample['num_edges'], 1])
        sample['link_utilization'] = tf.reshape(sample['link_utilization'][0:sample['num_edges']], [sample['num_edges'], 1])
        sample['bw_allocated'] = tf.reshape(sample['bw_allocated'][0:sample['num_edges']], [sample['num_edges'], 1])

        hiddenStates = tf.concat([sample['capacity'], sample['link_utilization'], sample['bw_allocated']], axis=1)

        paddings = tf.constant([[0, 0], [0, hparams['link_state_dim'] - len(listofDemands)]])
        link_state = tf.pad(tensor=hiddenStates, paddings=paddings, mode="CONSTANT")

        inputs = {'link_state': link_state, 'first': sample['first'][0:sample['length']],
                  'second': sample['second'][0:sample['length']], 'num_edges': sample['num_edges']}

        return inputs


def exec_MD_model_episodes(experience_memory, env, agent):
    MAX_link_utilization_MD = np.zeros(NUMBER_EPISODES)
    delay_MD = np.zeros(NUMBER_EPISODES)
    Utility_MD = np.zeros(NUMBER_EPISODES)
    iter_episode = 0
    count = 0
    isFlag = True
    while iter_episode < NUMBER_EPISODES:
        iter_step = 0
        utility = 0
        while iter_step < NUMBER_SESSIONS:
            demand = experience_memory[count][1]
            source = experience_memory[count][2]
            destination = experience_memory[count][3]
            if isFlag == True:
                state = env.eval_sap_reset(demand, source, destination)
            action, state_action = agent.choose_action(env, state, demand, source, destination)
            new_state, reward = env.step_eval(state, action, source, destination, demand)
            state = new_state
            iter_step += 1
            count += 1
            isFlag = False
            utility = math.log(sum(state[:, 2]*1000000), math.e) - math.log(sum((state[:, 2] / state[:, 0])), math.e)
            utility += utility
        isFlag = True

        eval_max_utilization = np.max(state[:, 1])
        MAX_link_utilization_MD[iter_episode] = eval_max_utilization
        delay = sum(state[:, 2] / (state[:, 0]))
        delay_MD[iter_episode] = delay
        Utility = utility
        Utility_MD[iter_episode] = Utility
        iter_episode += 1

    return MAX_link_utilization_MD, delay_MD, Utility_MD


class DDPGAgent(object):
    def __init__(self, action_dim, state_dim, action_bound, replay_buffer):
        self.replay_buffer = replay_buffer
        self.action_dim, self.state_dim, self.action_bound = action_dim, state_dim, action_bound
        self.noise = OUActionNoise(mu=np.zeros(action_dim))
        self.actor = DRLactor(self.action_dim)
        self.critic = DRLcritic()
        self.actor_target = self.actor
        self.critic_target = self.critic

        self.ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)

        self.actor_opt = tf.keras.optimizers.Adam(LR_A)
        self.critic_opt = tf.keras.optimizers.RMSprop(LR_C)

    def get_action(self, state):
        statecopy = np.copy(state)
        cmax = np.max(state[:, 0])
        statecopy[:, 0] = state[:, 0] / cmax
        bmax = np.max(listofDemands)
        statecopy[:, 2] = statecopy[:, 2] / bmax
        statecopy = tf.reshape(statecopy, (1, -1))
        action = self.actor(statecopy)
        noise = self.noise()
        action = action + np.clip(noise, 0, 10)
        return action


def exec_DRL_model_episodes(experience_memory, env, agent):
    MAX_link_utilization_DRL = np.zeros(NUMBER_EPISODES)
    delay_DRL = np.zeros(NUMBER_EPISODES)
    Utility_DRL = np.zeros(NUMBER_EPISODES)

    iter_episode = 0
    count = 0
    isFlag = True
    while iter_episode < NUMBER_EPISODES:
        iter_step = 0
        utility = 0
        while iter_step < NUMBER_SESSIONS:
            demand = experience_memory[count][1]
            source = experience_memory[count][2]
            destination = experience_memory[count][3]
            if isFlag == True:
                state = env.eval_sap_reset(demand, source, destination)
            action = agent.get_action(state)
            new_state, reward = env.step_eval(state, action, source, destination, demand)

            state = new_state
            iter_step += 1
            count += 1
            isFlag = False

            utility = math.log(sum(state[:, 2]*1000000), math.e) - math.log(sum((state[:, 2] / state[:, 0])), math.e)
            utility += utility
        isFlag = True

        eval_max_utilization = np.max(state[:, 1])
        MAX_link_utilization_DRL[iter_episode] = eval_max_utilization
        delay = sum(state[:, 2] / (state[:, 0]))
        delay_DRL[iter_episode] = delay
        Utility = utility
        Utility_DRL[iter_episode] = Utility
        iter_episode += 1

    return MAX_link_utilization_DRL, delay_DRL, Utility_DRL


def exec_ecmp_model_episodes(experience_memory):
    MAX_link_utilization_ecmp = np.zeros(NUMBER_EPISODES)
    delay_ecmp = np.zeros(NUMBER_EPISODES)
    Utility_ecmp = np.zeros(NUMBER_EPISODES)
    evaluate_ecmp = ecmp()
    iter_episode = 0
    count = 0
    isFlag = True
    link_capacities = []
    link_loads = []
    while iter_episode < NUMBER_EPISODES:
        iter_step = 0
        eval_max_utilization = eval_delay = utility = 0
        while iter_step < NUMBER_SESSIONS:
            demand = experience_memory[count][1]
            source = experience_memory[count][2]
            destination = experience_memory[count][3]
            if isFlag == True:
                state = env.eval_sap_reset(demand, source, destination)
                link_capacities = state[:, 0]
                link_loads = state[:, 2]

            max_utilization, delay, new_link_capacities, new_link_loads = evaluate_ecmp.eval_ecmp(source, destination,
                                    demand, link_capacities, link_loads)
            link_capacities = new_link_capacities
            link_loads = new_link_loads
            eval_max_utilization = max_utilization
            utility = math.log(sum(link_loads * 1000000), math.e) - math.log(delay, math.e)
            utility += utility
            eval_delay = delay
            iter_step += 1
            count += 1
            isFlag = False
        isFlag = True

        MAX_link_utilization_ecmp[iter_episode] = eval_max_utilization
        delay_ecmp[iter_episode] = eval_delay
        Utility = utility
        Utility_ecmp[iter_episode] = Utility

        iter_episode += 1

    return MAX_link_utilization_ecmp, delay_ecmp, Utility_ecmp


def exec_ospf_model_episodes(experience_memory):
    MAX_link_utilization_ospf = np.zeros(NUMBER_EPISODES)
    delay_ospf = np.zeros(NUMBER_EPISODES)
    Utility_ospf = np.zeros(NUMBER_EPISODES)
    evaluate_ospf = ospf('Ebone-Evaluate')
    iter_episode = 0
    count = 0
    isFlag = True
    link_capacities = []
    link_loads = []
    while iter_episode < NUMBER_EPISODES:
        iter_step = 0
        eval_max_utilization = eval_delay = utility =0
        while iter_step < NUMBER_SESSIONS:
            demand = experience_memory[count][1]
            source = experience_memory[count][2]
            destination = experience_memory[count][3]
            if isFlag == True:
                state = env.eval_sap_reset(demand, source, destination)
                link_capacities = state[:, 0]
                link_loads = state[:, 2]

            max_utilization, delay, new_link_capacities, new_link_loads = evaluate_ospf.eval_ospf(source, destination,
                        demand, link_capacities, link_loads, eval_max_utilization, eval_delay)

            link_capacities = new_link_capacities
            link_loads = new_link_loads
            eval_max_utilization = max_utilization
            eval_delay = delay
            utility = math.log(sum(link_loads * 1000000), math.e) - math.log(delay, math.e)
            utility += utility
            iter_step += 1
            count += 1
            isFlag = False
        isFlag = True
        MAX_link_utilization_ospf[iter_episode] = eval_max_utilization
        delay_ospf[iter_episode] = eval_delay
        Utility = utility
        Utility_ospf[iter_episode] = Utility

        iter_episode += 1

    return MAX_link_utilization_ospf, delay_ospf, Utility_ospf


if __name__ == "__main__":
    # python Evaluate.py -d ./Logs/Ebone/expsample_MPDRLAgentLogs.txt ./Logs/Ebone/expsample_DDPGAgentLogs.txt

    np.random.seed(SEED)
    env.seed(SEED)
    env.generate_environment(env, listofDemands)
    a_dim = env.num_links
    s_dim = np.array(env.graph_state.shape)
    action_bound = [0, 20]

    buffer = Replaybuffer(MEMORY_CAPACITY1)
    tf.compat.v1.enable_eager_execution()

    evaluate_ecmp = ecmp()
    evaluate_ospf = ospf('Ebone-Evaluate')

    parser = argparse.ArgumentParser(description='Parse file and create plots')
    parser.add_argument('-d', help='data file', type=str, required=True, nargs='+')
    args = parser.parse_args()
    aux = args.d[0].split(".")
    aux = aux[1].split("exp")
    differentiation_str = str(aux[1].split("Logs")[0])
    model_id = 0
    with open(args.d[0]) as fp:
        for line in reversed(list(fp)):
            arrayLine = line.split(":")
            if arrayLine[0] == 'MAX REWD':
                model_id = int(arrayLine[2].split(",")[0])
                break
    MD_agent = MDAgent(a_dim)
    checkpoint_dir = "./models/Ebone"+differentiation_str
    checkpoint = tf.train.Checkpoint(model1=MD_agent.actor, optimizer1=MD_agent.a_optimizer, model2=MD_agent.critic, optimizer2=MD_agent.c_optimizer)
    checkpoint.restore(checkpoint_dir + "/ckpt-" + str(model_id))
    print("Load model " + checkpoint_dir + "/ckpt-" + str(model_id))

    aux = args.d[1].split(".")
    aux = aux[1].split("exp")
    differentiation_str = str(aux[1].split("Logs")[0])
    model_id = 0
    with open(args.d[0]) as fp:
        for line in reversed(list(fp)):
            arrayLine = line.split(":")
            if arrayLine[0] == 'MAX REWD':
                model_id = int(arrayLine[2].split(",")[0])
                break
    DDPG_agent = DDPGAgent(a_dim, s_dim, action_bound, buffer)
    checkpoint_dir = "./models/Ebone" + differentiation_str
    checkpoint = tf.train.Checkpoint(model1=DDPG_agent.actor, optimizer1=DDPG_agent.actor_opt, model2=DDPG_agent.critic,
                                     optimizer2=DDPG_agent.critic_opt)
    checkpoint.restore(checkpoint_dir + "/ckpt-" + str(model_id))
    print("Load model " + checkpoint_dir + "/ckpt-" + str(model_id))

    experience_memory = deque(maxlen=NUMBER_EPISODES*NUMBER_SESSIONS)
    survival_time1 = np.zeros(NUMBER_EPISODES*NUMBER_SESSIONS)
    survival_time2 = np.zeros(NUMBER_EPISODES * NUMBER_SESSIONS)

    ep_num = 1
    average = 10

    while ep_num <= NUMBER_EPISODES:
        i = 0
        demand = stats.poisson.rvs(mu=average, size=NUMBER_SESSIONS, random_state=None)
        while i < NUMBER_SESSIONS:
            source = np.random.choice(env.nodes)
            # We pick a pair of SOURCE,DESTINATION different nodes
            destination = np.random.choice(env.nodes)
            while True:
                destination = np.random.choice(env.nodes)
                if destination != source:
                    experience_memory.append((ep_num, demand[i], source, destination))
                    break
            i += 1
            
        average += 5
        ep_num += 1

    MAX_link_utilization_ecmp, delay_ecmp, Utility_ecmp = exec_ecmp_model_episodes(experience_memory)
    MAX_link_utilization_ospf, delay_ospf, Utility_ospf = exec_ospf_model_episodes(experience_memory)
    MAX_link_utilization_MD, delay_MD, Utility_MD = exec_MD_model_episodes(experience_memory, env, MD_agent)
    MAX_link_utilization_DDPG, delay_DDPG, Utility_DDPG = exec_DRL_model_episodes(experience_memory, env, DDPG_agent)

    if not os.path.exists("./evaluate_result/Ebone"):
        os.makedirs("./evaluate_result/Ebone")
    file = open("./evaluate_result/Ebone/MAX_link_utilization.csv", "a")
    writer = csv.writer(file)
    with open("./evaluate_result/Ebone/MAX_link_utilization.csv", "r") as csv_reader:
        reader = csv.reader(csv_reader)
        if not [row for row in reader]:
            writer.writerow(['10', '15', '20', '25', '30', '35', '40'])
            writer.writerow(MAX_link_utilization_MD)
            writer.writerow(MAX_link_utilization_DDPG)
            writer.writerow(MAX_link_utilization_ecmp)
            writer.writerow(MAX_link_utilization_ospf)

    file = open("./evaluate_result/Ebone/delay.csv", "a")
    writer = csv.writer(file)
    with open("./evaluate_result/Ebone/delay.csv", "r") as csv_reader:
        reader = csv.reader(csv_reader)
        if not [row for row in reader]:
            writer.writerow(['10', '15', '20', '25', '30', '35', '40'])
            writer.writerow(delay_MD)
            writer.writerow(delay_DDPG)
            writer.writerow(delay_ecmp)
            writer.writerow(delay_ospf)

    file = open("./evaluate_result/Ebone/Utility.csv", "a")
    writer = csv.writer(file)
    with open("./evaluate_result/Ebone/Utility.csv", "r") as csv_reader:
        reader = csv.reader(csv_reader)
        if not [row for row in reader]:
            writer.writerow(['10', '15', '20', '25', '30', '35', '40'])
            writer.writerow(Utility_MD)
            writer.writerow(Utility_DDPG)
            writer.writerow(Utility_ecmp)
            writer.writerow(Utility_ospf)

    file.flush()
    file.close()

    gc.collect()

    if not os.path.exists("./img/Ebone"):
        os.makedirs("./img/Ebone")
    if not os.path.exists("./img/Sprintlink"):
        os.makedirs("./img/Sprintlink")
    if not os.path.exists("./img/Tiscali"):
        os.makedirs("./img/Tiscali")

    x = [10, 15, 20, 25, 30, 35, 40]
    u1 = MAX_link_utilization_ecmp
    plt.plot(x, u1, 'g', label="ECMP", marker='^', linewidth=2)
    u2 = MAX_link_utilization_ospf
    plt.plot(x, u2, 'b', label="OSPF", marker='o', linewidth=2)
    u3 = MAX_link_utilization_DDPG
    plt.plot(x, u3, 'm', label="DDPG", marker='+', linewidth=2)
    u4 = MAX_link_utilization_MD
    plt.plot(x, u4, 'r', label="MPDRL", marker='*', linewidth=2)
    plt.xlabel("Traffic Demand(Mbps)")
    plt.ylabel("Maximum Link Utilization")
    plt.ylim(0, 1)
    plt.legend()
    # plt.savefig("./img/Ebone/Maximum-Link-Utilization.pdf")
    plt.show()

    d1 = delay_ecmp
    plt.plot(x, d1, 'g', label="ECMP", marker='^', linewidth=2)
    d2 = delay_ospf
    plt.plot(x, d2, 'b', label="OSPF", marker='o', linewidth=2)
    d3 = delay_DDPG
    plt.plot(x, d3, 'm', label="DDPG", marker='+', linewidth=2)
    d4 = delay_MD
    plt.plot(x, d4, 'r', label="MPDRL", marker='*', linewidth=2)
    plt.xlabel("Traffic Demand(Mbps)")
    plt.ylabel("End-to-end Delay")
    plt.legend()
    # plt.savefig("./img/Ebone/Delay.pdf")
    plt.show()

    y1 = Utility_ecmp
    plt.plot(x, y1, 'g', label="ECMP", marker='^', linewidth=2)
    y2 = Utility_ospf
    plt.plot(x, y2, 'b', label="OSPF", marker='o', linewidth=2)
    y3 = Utility_DDPG
    plt.plot(x, y3, 'm', label="DDPG", marker='+', linewidth=2)
    y4 = Utility_MD
    plt.plot(x, y4, 'r', label="MPDRL", marker='*', linewidth=2)
    plt.xlabel("Traffic Demand(Mbps)")
    plt.ylabel("Utility")
    plt.legend()
    # plt.savefig("./img/Ebone/Utility.pdf")
    plt.show()
