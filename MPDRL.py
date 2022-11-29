import gc
import os
import random
import numpy as np
import tensorflow as tf
from collections import deque
from Actor import Actor
from Critic import Critic
from OU import OUActionNoise
from ReplayBuffer import ReplayBuffer
from Environment import Env
import matplotlib.pyplot as plt
import time
gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

env = Env('Ebone')
MAX_EPISODES = 200
MAX_EP_STEPS = 10

SEED = 20
ep_reward_list = []
hparams = {
    'l2': 0.1,
    'dropout_rate': 0.01,
    'link_state_dim': 20,
    'readout_units': 38,
    'LR_A': 0.000001,  # learning rate for actor
    'batch_size': 16,
    'T': 4
}

LR_C = 0.00001  # learning rate for critic
GAMMA = 0.99  # discount rate
TAU = 0.01  # soft update rate
MEMORY_CAPACITY = 10000
BATCH_SIZE = 16
MAX_QUEUE_SIZE = 1000

listofDemands = [10, 25, 40]

differentiation_str = "sample_MPDRLAgent"
checkpoint_dir = "./models/Ebone" + differentiation_str

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


class MPDRLAgent(object):
    def __init__(self, a_dim, s_dim, a_bound):
        self.pointer = 0
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound
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

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)
        self.pointer += env.graph_state.shape[0]

    def choose_action(self, env, state, demand, source, destination):
        # List of graphs
        listGraphs = []
        # List of graph features that are used in the cummin() call
        list_k_features = list()
        pathList = env.allPaths[str(source) + ':' + str(destination)]
        path = np.random.randint(0, len(pathList))
        while path < len(pathList):
            state_copy = np.copy(state)
            currentPath = pathList[path]
            i = 0
            j = 1

            while j < len(currentPath):
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
        cmin = np.min(copyGraph[:, 0])
        self.capacity_feature = (copyGraph[:, 0] - cmin) / (cmax - cmin)
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

    @tf.function
    def _forward_pass(self, x):
        prediction_state = self.actor(x[0], x[1], x[2], x[3], x[4], training=True)
        preds_next_target = tf.stop_gradient(self.target_actor(x[6], x[7], x[9], x[10], x[11], training=True))
        return prediction_state, preds_next_target

    def ema_update(self):
        """
        Soft updating by exponential smoothing
        """
        paras = self.actor.trainable_weights + self.critic.trainable_weights
        self.ema.apply(paras)
        for m, n in zip(self.target_actor.trainable_weights + self.target_critic.trainable_weights, paras):
            m.assign(self.ema.average(n))

    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return
        a_batch = random.sample(self.a_memory, batch_size)
        state, action, reward, new_state, done = self.memory.sample(BATCH_SIZE)
        bs = state
        ba = action
        br = reward
        bs_ = new_state

        with tf.GradientTape() as tape:
            preds_newstate = []
            for x in a_batch:
                prediction_state, preds_next_target = self._forward_pass(x)
                preds_newstate.append(preds_next_target[0])
            a_newstate = tf.stack(preds_newstate, axis=1)
            a_newstate = tf.reshape(a_newstate, [a_newstate.shape[0]*batch_size, -1])
            bs_ = tf.convert_to_tensor(bs_)
            bs_ = tf.cast(bs_, tf.float32)
            i = 0
            y = []
            q_ = []
            q = []
            while i < BATCH_SIZE:
                input1 = tf.concat([bs_[a_dim*i:a_dim*(i+1)], a_newstate[a_dim*i:a_dim*(i+1)]], axis=1)
                input2 = tf.concat([bs[a_dim*i:a_dim*(i+1)], ba[a_dim*i:a_dim*(i+1)]], axis=1)
                q_.append(self.target_critic(input1))
                y.append(br[a_dim*i:a_dim*(i+1)] + GAMMA * q_[i])
                q.append(self.critic(input2))
                i = i + 1
            td_error = tf.losses.mean_squared_error(y, q)
        c_grads = tape.gradient(td_error, self.critic.trainable_weights)
        self.c_optimizer.apply_gradients(zip(c_grads, self.critic.trainable_weights))
        del tape

        preds_state = []
        target = []
        with tf.GradientTape() as tape:
            for x in a_batch:
                prediction_state, preds_next_target = self._forward_pass(x)
                target.append(tf.stop_gradient([x[5] + GAMMA*tf.math.reduce_max(preds_next_target)*(1-x[8])]))
                preds_state.append(prediction_state[0])
            a_state = tf.stack(preds_state, axis=1)
            a_state = tf.reshape(a_state, [a_state.shape[0] * batch_size, -1])
            i = 0
            q = []
            while i < BATCH_SIZE:
                input3 = tf.concat([bs[a_dim*i:a_dim*(i+1)], a_state[a_dim*i:a_dim*(i+1)]], axis=1)
                q.append(self.critic(input3))
                i = i + 1
            loss = tf.keras.losses.MSE(tf.stack(target, axis=1), tf.stack(preds_state, axis=1))
            regularization_loss = sum(self.actor.losses)
            loss = loss + regularization_loss
            a_loss = -tf.reduce_mean(q)+loss
        a_grads = tape.gradient(a_loss, self.actor.trainable_weights)
        self.a_optimizer.apply_gradients(zip(a_grads, self.actor.trainable_weights))
        self.ema_update()
        gc.collect()

    def add_sample(self, env, state_action, action, reward, new_state, new_demand, new_source,
                   new_destination, done):
        self.bw_allocated_feature.fill(0.0)
        new_state_copy = np.copy(new_state)
        listGraphs = []
        list_k_features = list()
        state_action['graph_id'] = tf.fill([tf.shape(state_action['link_state'])[0]], 0)

        pathList = env.allPaths[str(new_source) +':'+ str(new_destination)]
        path = np.random.randint(0, len(pathList))
        while path < len(pathList):
            state_copy = np.copy(state)
            currentPath = pathList[path]
            i = 0
            j = 1
            while (j < len(currentPath)):
                new_state_copy[env.edgesDict[str(currentPath[i]) + ':' + str(currentPath[j])]][1] = new_demand
                i = i + 1
                j = j + 1
            listGraphs.append(state_copy)
            features = self.get_graph_features(env, state_copy)
            list_k_features.append(features)
            new_state_copy[:, 1] = 0
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
        self.a_memory.append((state_action['link_state'], state_action['graph_id'], state_action['first'],
                            state_action['second'], tf.convert_to_tensor(state_action['num_edges']),
                            tf.convert_to_tensor(reward, dtype=tf.float32), tensors['link_state'], tensors['graph_id'],
                            tf.convert_to_tensor(int(done == True), dtype=tf.float32), tensors['first'], tensors['second'],
                            tf.convert_to_tensor(tensors['num_edges'])))  # 12

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()


if __name__ == "__main__":
    np.random.seed(SEED)
    env.seed(SEED)
    env.generate_environment(env, listofDemands)

    s_dim = env.graph_state.shape
    a_dim = env.num_links
    a_bound = [0, 20]
    tf.compat.v1.enable_eager_execution()
    agent = MPDRLAgent(a_dim, s_dim, a_bound)

    batch_size = hparams['batch_size']
    counter_store_model = 1
    if not os.path.exists("./Logs"):
        os.makedirs("./Logs")

    fileLogs = open("./Logs/Ebone/exp" + differentiation_str + "Logs.txt", "a")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(model1=agent.actor, optimizer1=agent.a_optimizer, model2=agent.critic, optimizer2=agent.c_optimizer)

    var = 3
    start = time.time()
    for i in range(MAX_EPISODES):
        state, demand, source, destination = env.reset()
        ep_reward = 0
        j = 1
        while j <= MAX_EP_STEPS:
            a, state_action = agent.choose_action(env, state, demand, source, destination)
            a = np.clip(a, *a_bound)
            new_state, reward, new_source, new_destination, new_demand, done = env.step(state, a, source, destination, demand)
            agent.remember(state, a, reward, new_state, True)
            agent.add_sample(env, state_action, a, reward, new_state, new_demand, new_source, new_destination, done)
            if agent.pointer > MEMORY_CAPACITY:
                var *=.9995
                agent.learn()

            state = new_state
            demand = new_demand
            source = new_source
            destination = new_destination
            ep_reward += reward
            if j == MAX_EP_STEPS:
                print('Episode:', i, ' Steps: %2d' % j, ' Reward: %7.2f' % ep_reward)
                ep_reward_list = np.append(ep_reward_list, ep_reward)
                break
            j = j + 1
        env.graph_state[:, 2] = 0

    end = time.time()
    print('Time:', (end-start)/60)

    evalMeanReward = np.mean(ep_reward_list)
    max_reward = np.max(ep_reward_list)
    checkpoint.save(checkpoint_prefix)
    fileLogs.write(">," + str(evalMeanReward) + ",\n")
    fileLogs.write("MAX REWD: " + str(max_reward) + " MODEL_ID: " + str(counter_store_model) + ",\n")
    fileLogs.flush()
    fileLogs.close()
    gc.collect()
    if not os.path.exists("./img/"):
        os.makedirs("./img")
    plt.plot(ep_reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig("./img/MPDRL-Ebone.pdf")
    plt.show()

