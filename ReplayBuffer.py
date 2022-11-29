import random
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def store_transition(self, state, action, reward, new_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        action_ = action.transpose()
        R = np.empty(action_.shape)
        R[:] = reward
        d = np.empty(action_.shape)
        d[:] = done
        self.buffer[self.position] = np.hstack((state, action_, R, new_state, d))
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state_batch = np.empty(shape=(0, 3))
        action_batch = np.empty(shape=(0, 1))
        reward_batch = np.empty(shape=(0, 1))
        new_state_batch = np.empty(shape=(0, 3))
        done_batch = np.empty(shape=(0, 1))

        i = 0
        while i < batch_size:
            b1 = batch[i][:, 0:3]
            state_batch = np.append(state_batch, b1, axis=0)
            b2 = batch[i][:, 3:4]
            action_batch = np.append(action_batch, b2, axis=0)
            b3 = batch[i][:, 4:5]
            reward_batch = np.append(reward_batch, b3, axis=0)
            b4 = batch[i][:, 5:8]
            new_state_batch = np.append(new_state_batch, b4, axis=0)
            b5 = batch[i][:, 8:9]
            done_batch = np.append(done_batch, b5, axis=0)
            i = i+1

        return state_batch, action_batch, reward_batch, new_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)







