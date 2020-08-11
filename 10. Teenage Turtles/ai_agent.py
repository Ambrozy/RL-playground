import retro
import random

env = retro.make(game='TeenageMutantNinjaTurtlesIIITheManhattanProject-Nes', scenario='scenario.json', info='data.json')
obs = env.reset()
n_actions = env.action_space.n                  # 9
n_observation = env.observation_space.shape     # (224, 240, 3)
buttons = env.buttons                           # ['B', None, 'Select', 'Start', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A']
buttons_combos = env.button_combos              # [[0, 16, 32], [0, 64, 128], [0, 1, 256, 257]]


class Agent:
    def __init__(self, env):
        self.env = env

    def sample(self):
        if random.random() > .2:
            return self.env.action_space.sample()
        return [0, 0, 0, 0, 0, 0, 0, 1, 0]


class State:
    def __init__(self, env):
        self.env = env
        self.last_health = 16

    def step(self, action):
        obs, rew, done, info = self.env.step(action)

        one_health_sum = 4216
        info['health'] = int(obs[207:214, 47:111, 0:1].sum() / one_health_sum)
        delta_health = info['health'] - self.last_health
        if delta_health < 0:
            rew += delta_health * 50
        self.last_health = info['health']

        return obs, rew, done, info


agent = Agent(env)
state = State(env)
while True:
    action = agent.sample()
    # score +100 => rew = +100
    # lives -1 => rew = -50000
    # health -1 => rew = -50
    obs, rew, done, info = state.step(action)
    print(rew, info, action)
    env.render()
    if done:
        obs = env.reset()

env.close()
