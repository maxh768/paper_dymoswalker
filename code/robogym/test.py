import gym, robo_gym

env = gym.make('EmptyEnvironmentURSim-v0', ip='<127.0.0.1>')
env.reset()

env = gym.make('EmptyEnvironmentURSim-v0', ip='<127.0.0.1>', gui=True)