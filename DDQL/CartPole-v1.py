# References:
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# https://towardsdatascience.com/deep-q-learning-for-the-cartpole-44d761085c2f

import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

env = gym.make("CartPole-v1")

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


######################################################################
# Memory
######################################################################
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


######################################################################
# Model
######################################################################
class DQN(nn.Module):

    def __init__(self, observation_space, action_space):
        super(DQN, self).__init__()
        self.l1 = nn.Linear(observation_space, 256)
        self.head = nn.Linear(256, action_space)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.l1(x))
        return self.head(x)


######################################################################
# Training
######################################################################
BATCH_SIZE = 512
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10  # each 10 episodes copy policy_net params to target_net

initial_state = env.reset()
n_actions = env.action_space.n
n_observations = env.observation_space.shape[0]

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)

steps_done = 0


def select_action(state):  # return tensor([[0]]) or tensor([[1]])
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # Pick action with the larger expected reward.
            policy_net.eval()
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


######################################################################
# Visualization
######################################################################
episode_durations = []


def moving_average(a, n=10):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return np.concatenate([
        ret[:n-1] / np.arange(1, n),
        ret[n-1:] / n,
    ])


def plot_durations(title='Training...'):
    plt.figure(1)
    plt.clf()
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(episode_durations, label='Episode duration')
    # Take 100 episode averages and plot them too
    if len(episode_durations) >= 2:
        n = np.min([len(episode_durations), 50])
        means = moving_average(episode_durations, n=n)
        plt.plot(means, label='Mean episode duration')

    plt.legend(loc='upper left')
    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)


######################################################################
# Training loop
######################################################################
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)  # [[0], [1], [0], [1], ...]
    reward_batch = torch.cat(batch.reward)  # [1, 1, 1, ...]

    policy_net.train()
    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken.
    state_action_values = policy_net(state_batch).gather(1, action_batch)  # [[.2], [.3], ...]

    # Compute V(s_{t+1}) for all next states. Computed based on the "older" target_net
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()  # [.2, .3, ...]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch  # [1.02, 1.03, ...]

    # Compute Huber loss
    # F.smooth_l1_loss([.2, .3, ...], [1.02, 1.03, ...])
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)  # clamp_ = -2, -1, 0, 1, 2 => -1, -1, 0, 1, 1
    optimizer.step()


def update():
    target_net.load_state_dict(policy_net.state_dict())


def soft_update(TAU=0.1):
    model_params = policy_net.named_parameters()
    target_params = target_net.named_parameters()

    updated_params = dict(target_params)

    for model_name, model_param in model_params:
        if model_name in target_params:
            # Update parameter
            updated_params[model_name].data.copy_(TAU * model_param.data + (1 - TAU) * target_params[model_param].data)

    target_net.load_state_dict(updated_params)


num_episodes = 300
for i_episode in range(num_episodes):
    # Initialize the environment and state
    state = env.reset()
    state = torch.tensor(state, device=device, dtype=torch.float).unsqueeze(0)
    for t in count():
        # Select and perform an action
        action = select_action(state)
        next_state, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)  # [1.0]

        # Observe new state
        if not done:
            next_state = torch.tensor(next_state, device=device, dtype=torch.float).unsqueeze(0)
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        update()

env.close()
torch.save(target_net.state_dict(), 'model-dql.pt')

plot_durations('Complete')
plt.show()
