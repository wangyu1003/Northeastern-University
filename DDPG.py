import gc
import os
import random
from OU import OUActionNoise
from Environment import Env
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense

env = Env('Ebone')
listofDemands = [10, 25, 40]

MAX_EPISODES = 200
MAX_EP_STEPS = 10

SEED = 8
ep_reward_list = []
aloss_list = []

LR_A = 0.000001  # learning rate for actor
LR_C = 0.00001  # learning rate for critic
GAMMA = 0.99  # reward discount
TAU = 0.01  # soft replacement
MEMORY_CAPACITY = 1000  # size of replay buffer
BATCH_SIZE = 16 # update action batch size

os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(1)
differentiation_str = "sample_DDPGAgent"
checkpoint_dir = "./models/Ebone" + differentiation_str


class Replaybuffer:

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class DRLactor(tf.keras.Model):
    def __init__(self, action_dim):
        super(DRLactor, self).__init__()
        W_init = tf.random_normal_initializer(mean=0, stddev=0.3)
        b_init = tf.constant_initializer(0.1)
        # Define layers here
        self.flatten = tf.keras.layers.Flatten()
        self.layer1 = Dense(128, activation=tf.nn.relu, kernel_initializer=W_init, bias_initializer=b_init, name='A_l1')
        self.layer2 = Dense(64, activation=tf.nn.relu, kernel_initializer=W_init, bias_initializer=b_init, name='A_l2')
        self.a = Dense(action_dim, activation=tf.nn.relu, name='A_a')

    @tf.function
    def __call__(self, state_input):
        x = self.flatten(state_input)
        action1 = self.layer1(x)
        action2 = self.layer2(action1)
        action = self.a(action2)
        return action


class DRLcritic(keras.Model):
    def __init__(self):
        super(DRLcritic, self).__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = Dense(128, activation=tf.nn.leaky_relu)
        self.fc2 = Dense(64, activation=tf.nn.leaky_relu)
        self.q = Dense(1, activation=None)

    @tf.function
    def __call__(self, state_input, action_input):
        x = self.flatten(state_input)
        y = self.flatten(action_input)
        action_value = self.fc1(tf.concat([x, y], axis=1))
        action_value = self.fc2(action_value)
        q = self.q(action_value)
        return q


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

    def ema_update(self):
        paras = self.actor.trainable_weights + self.critic.trainable_weights
        self.ema.apply(paras)
        for i, j in zip(self.actor_target.trainable_weights + self.critic_target.trainable_weights, paras):
            i.assign(self.ema.average(j))

    def get_action(self, state):
        statecopy = np.copy(state)
        cmax = np.max(state[:, 0])
        cmin = np.min(state[:, 0])
        statecopy[:, 0] = (state[:, 0] - cmin) / (cmax - cmin)
        bmax = np.max(listofDemands)
        statecopy[:, 2] = statecopy[:, 2] / bmax

        statecopy = tf.reshape(statecopy, (1, -1))
        action = self.actor(statecopy)
        noise = self.noise()
        action = action + np.clip(noise, 0, 10)
        return action

    def learn(self):
        states, actions, rewards, states_, done = self.replay_buffer.sample(BATCH_SIZE)
        rewards = rewards[:, np.newaxis]
        done = done[:, np.newaxis]

        with tf.GradientTape() as tape:
            actions_ = self.actor_target(states_)
            q_ = self.critic_target(states_, actions_)
            target = rewards + (1 - done) * GAMMA * q_
            q_pred = self.critic(states, actions)
            td_error = tf.losses.mean_squared_error(target, q_pred)

        critic_grads = tape.gradient(td_error, self.critic.trainable_weights)
        self.critic_opt.apply_gradients(zip(critic_grads, self.critic.trainable_weights))

        with tf.GradientTape() as tape:
            actions = self.actor(states)
            q = self.critic(states, actions)
            actor_loss = -tf.reduce_mean(q)

        actor_grads = tape.gradient(actor_loss, self.actor.trainable_weights)
        self.actor_opt.apply_gradients(zip(actor_grads, self.actor.trainable_weights))
        self.ema_update()

        return actor_loss


if __name__ == '__main__':
    np.random.seed(SEED)
    env.seed(SEED)
    env.generate_environment(env, listofDemands)

    state_dim = np.array(env.graph_state.shape)
    action_dim = env.num_links
    action_bound = [0, 20]
    var = 3
    buffer = Replaybuffer(MEMORY_CAPACITY)
    agent = DDPGAgent(action_dim, state_dim, action_bound, buffer)
    rewards_test = np.zeros(MAX_EPISODES)

    if not os.path.exists("./Logs/Ebone"):
        os.makedirs("./Logs/Ebone")
    fileLogs = open("./Logs/Ebone/exp" + differentiation_str + "Logs.txt", "a")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(model1=agent.actor, optimizer1=agent.actor_opt, model2=agent.critic, optimizer2=agent.critic_opt)
    counter_store_model = 1
    for i in range(MAX_EPISODES):
        state, demand, source, destination = env.reset()
        ep_reward = 0
        j = 1
        while j <= MAX_EP_STEPS:
            action = agent.get_action(state)
            action = np.clip(action, *action_bound)
            new_state, reward, new_source, new_destination, new_demand, done = env.step(state, action, source, destination, demand)

            done = 1 if done is True else 0
            buffer.push(state, action, reward, new_state, done)
            if len(buffer) >= MEMORY_CAPACITY:
                var *= .9995
                actor_loss = agent.learn()
                aloss_list = np.append(aloss_list, actor_loss)

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

    evalMeanReward = np.mean(ep_reward_list)
    max_reward = np.max(ep_reward_list)

    checkpoint.save(checkpoint_prefix)
    fileLogs.write(">," + str(evalMeanReward) + ",\n")
    fileLogs.write("MAX REWD: " + str(max_reward) + " MODEL_ID: " + str(counter_store_model) + ",\n")
    fileLogs.flush()
    fileLogs.close()
    gc.collect()

    plt.plot(aloss_list)
    plt.xlabel("Episode")
    plt.ylabel("a_loss")
    plt.show()
    plt.plot(ep_reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.show()

