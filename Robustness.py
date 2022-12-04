import argparse
import gc
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
import matplotlib.pyplot as plt
import csv

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

env = Env('Ebone')

SEED = 20   #

NUMBER_EPISODES = 6
NUMBER_SESSIONS = 14

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
MEMORY_CAPACITY1 = 2000
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
        }
        )

        a_predict = self.actor(tensors['link_state'], tensors['graph_id'], tensors['first'],
                                                tensors['second'], tensors['num_edges'], training=False).numpy()
        noise = self.noise()
        action_ = a_predict + np.clip(noise, 0, 10)

        return action_, list_k_features[0]

    def get_graph_features(self, env, copyGraph):
        cmax = np.max(copyGraph[:, 0])
        self.capacity_feature = copyGraph[:, 0]/cmax
        self.bw_allocated_feature = (copyGraph[:, 2]) / demand

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


def exec_MD(experience_memory, link_memory, env, agent):
    iter_step = 1
    counts = 0
    record = deque(maxlen=12)
    times = 1
    while iter_step <= NUMBER_SESSIONS:
        demand = experience_memory[iter_step-1][0]
        source = experience_memory[iter_step-1][1]
        destination = experience_memory[iter_step-1][2]
        if iter_step == 1:
            state = env.eval_sap_reset(demand, source, destination)
            for position in link_memory:
                state[position][0] = 0
        action, state_action = agent.choose_action(env, state, demand, source, destination)
        new_state, reward = env.step_robust(state, action, source, destination, demand)
        state = new_state
        iter_step += 1
        flag_count = True
        k = 0
        while k < len(link_memory):
              position = link_memory[k]
              if state[position][0] < 0:
                  i = 0
                  if iter_step == 1:
                      record.append([position, 1])
                  flag = True
                  while i < len(record):
                      if record[i][0] == position:
                          record[i][1] += 1
                          times = record[i][1]
                          flag = False
                          break
                      i += 1
                  if flag:
                      times = 1
                      record.append([position, 1])
                  if state[position][2] == demand * times and flag_count:
                      flag_count = False
                      counts += 1
            k += 1
    return counts


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
        statecopy[:, 2] = statecopy[:, 2] / demand
        statecopy = tf.reshape(statecopy, (1, -1))
        action = self.actor(statecopy)
        noise = self.noise()
        action = action + np.clip(noise, 0, 10)
        return action


def exec_DDPG(experience_memory, link_memory, time_interval, env, agent):
    iter_step = 1
    counts = 0
    record = deque(maxlen=12)
    times = 1
    while iter_step <= NUMBER_SESSIONS:
        demand = experience_memory[iter_step - 1][0]
        source = experience_memory[iter_step - 1][1]
        destination = experience_memory[iter_step - 1][2]
        if iter_step == 1:
            state = env.eval_sap_reset(demand, source, destination)
            for position in link_memory:
                state[position][0] = 0
        action = agent.get_action(state)
        new_state, reward = env.step_robust(state, action, source, destination, demand)
        state = new_state
        iter_step += 1
        flag_count = True
        k = 0
        while k < len(link_memory):
              position = link_memory[k]
              if state[position][0] < 0:
                  i = 0
                  if iter_step == 1:
                      record.append([position, 1])
                  flag = True
                  while i < len(record):
                      if record[i][0] == position:
                          record[i][1] += 1
                          times = record[i][1]
                          flag = False
                          break
                      i += 1
                  if flag:
                      times = 1
                      record.append([position, 1])
                  if state[position][2] == demand * times and flag_count:
                      flag_count = False
                      counts += 1
            k += 1
    return counts


def exec_ECMP(experience_memory, link_memory):
    evaluate_ecmp = ecmp()
    iter_step = 1
    counts = 0
    link_capacities = []
    link_loads = []
    while iter_step <= NUMBER_SESSIONS:
        demand = experience_memory[iter_step - 1][0]
        source = experience_memory[iter_step - 1][1]
        destination = experience_memory[iter_step - 1][2]
        i = 0
        if iter_step == 1:
            state = env.eval_sap_reset(demand, source, destination)
            link_capacities = state[:, 0]
            link_loads = state[:, 2]
            for position in link_memory:
                link_capacities[position] = 0
        new_link_capacities, new_link_loads, flag = evaluate_ecmp.robust_ecmp(source, destination, demand, link_capacities, link_loads, link_memory)
        link_capacities = new_link_capacities
        link_loads = new_link_loads
        iter_step += 1
        if flag:
            counts += 1

    return counts


def exec_OSPF(experience_memory, link_memory):
    evaluate_ospf = ospf('Ebone-Evaluate')
    iter_step = 1
    counts = 0
    record = deque(maxlen=12)
    times = 1
    link_capacities = []
    link_loads = []
    while iter_step <= NUMBER_SESSIONS:
        demand = experience_memory[iter_step - 1][0]
        source = experience_memory[iter_step - 1][1]
        destination = experience_memory[iter_step - 1][2]
        if iter_step == 1:
            state = env.eval_sap_reset(demand, source, destination)
            link_capacities = state[:, 0]
            link_loads = state[:, 2]
            for position in link_memory:
                link_capacities[position] = 0

        new_link_capacities, new_link_loads = evaluate_ospf.robust_ospf(source, destination, demand, link_capacities, link_loads)
        link_capacities = new_link_capacities
        link_loads = new_link_loads

        iter_step += 1
        flag_count = True
        k = 0
        while k < len(link_memory):
              position = link_memory[k]
              if link_capacities[position] < 0:
                  i = 0
                  if iter_step == 1:
                      record.append([position, 1])
                  flag = True
                  while i < len(record):
                      if record[i][0] == position:
                          record[i][1] += 1
                          times = record[i][1]
                          flag = False
                          break
                      i += 1
                  if flag:
                      times = 1
                      record.append([position, 1])
                  if link_loads[position] == demand * times and flag_count:
                      flag_count = False
                      counts += 1
            k += 1

    return counts

if __name__ == "__main__":
    #  python Robustness.py -d ./Logs/Ebone/expsample_MPDRLAgentLogs.txt ./Logs/Ebone/expsample_DDPGAgentLogs.txt
    np.random.seed(SEED)
    env.seed(SEED)
    demand = 40
    env.generate_environment(env, demand)
    a_dim = env.num_links
    s_dim = np.array(env.graph_state.shape)
    action_bound = [0, 20]

    buffer = Replaybuffer(MEMORY_CAPACITY1)
    tf.compat.v1.enable_eager_execution()

    MD_counts = []
    DDPG_counts = []
    ECMP_counts = []
    OSPF_counts = []

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
    checkpoint_dir = "./models/Ebone" + differentiation_str
    checkpoint = tf.train.Checkpoint(model1=MD_agent.actor, optimizer1=MD_agent.a_optimizer, model2=MD_agent.critic,
                                     optimizer2=MD_agent.c_optimizer)

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

    experience_memory = deque(maxlen=NUMBER_SESSIONS)
    ep_num = 1

    while ep_num <= NUMBER_SESSIONS:
        source = np.random.choice(env.nodes)
        destination = np.random.choice(env.nodes)
        while True:
            destination = np.random.choice(env.nodes)
            if destination != source:
                experience_memory.append((demand, source, destination))
                break
        ep_num += 1

    iteration = 1
    nums = 2
    while iteration <= NUMBER_EPISODES:
        link_memory = deque(maxlen=nums)
        j = 0
        index = set()
        while len(index) < nums:
            index.add(random.randint(0, env.num_links-1))
        arr = []
        for m in index:
            arr.append(m)
        while j < nums:
            link_memory.append(arr[j])
            j += 1
        iteration += 1
        nums += 2
        counts1 = exec_MD(experience_memory, link_memory, env, MD_agent)
        counts2 = exec_DDPG(experience_memory, link_memory, env, DDPG_agent)
        counts3 = exec_ECMP(experience_memory, link_memory)
        counts4 = exec_OSPF(experience_memory, link_memory)
        MD_counts.append(counts1)
        DDPG_counts.append(counts2)
        ECMP_counts.append(counts3)
        OSPF_counts.append(counts4)
        link_memory.clear()
    print(MD_counts, DDPG_counts, ECMP_counts, OSPF_counts)

    if not os.path.exists("./Path_failure_result"):
        os.makedirs("./Path_failure_result")
    file = open("./Path_failure_result/Ebone_4MD.csv", "a")
    writer = csv.writer(file)
    with open("./Path_failure_result/Ebone_4MD.csv", "r") as csv_reader:
        reader = csv.reader(csv_reader)
        if not [row for row in reader]:
            writer.writerow(['2', '4', '6', '8', '10', '12'])
            writer.writerow(MD_counts)
        else:
            writer.writerow(MD_counts)
    file = open("./Path_failure_result/Ebone_3DDPG.csv", "a")
    writer = csv.writer(file)
    with open("./Path_failure_result/Ebone_3DDPG.csv", "r") as csv_reader:
        reader = csv.reader(csv_reader)
        if not [row for row in reader]:
            writer.writerow(['2', '4', '6', '8', '10', '12'])
            writer.writerow(DDPG_counts)
        else:
            writer.writerow(DDPG_counts)
    file = open("./Path_failure_result/Ebone_2ECMP.csv", "a")
    writer = csv.writer(file)
    with open("./Path_failure_result/Ebone_2ECMP.csv", "r") as csv_reader:
        reader = csv.reader(csv_reader)
        if not [row for row in reader]:
            writer.writerow(['2', '4', '6', '8', '10', '12'])
            writer.writerow(ECMP_counts)
        else:
            writer.writerow(ECMP_counts)
    file = open("./Path_failure_result/Ebone_1OSPF.csv", "a")
    writer = csv.writer(file)
    with open("./Path_failure_result/Ebone_1OSPF.csv", "r") as csv_reader:
        reader = csv.reader(csv_reader)
        if not [row for row in reader]:
            writer.writerow(['2', '4', '6', '8', '10', '12'])
            writer.writerow(OSPF_counts)
        else:
            writer.writerow(OSPF_counts)

    file.flush()
    file.close()
    gc.collect()

