import random
import gymnasium as gym
import hydra
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from torch.utils.tensorboard import SummaryWriter

def make_env(env_name, env_kwargs: dict = {}, wrappers: list = []):
    env = gym.make(env_name, **env_kwargs)
    for wrapper in wrappers:
        env = hydra.utils.instantiate(wrapper, env=env)
    return env

def anim(frames, titles=None, filename=None, show=True):
    plt.figure(figsize=(frames[0].shape[1], frames[0].shape[0]), dpi=144)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])
        if titles is not None:
            plt.title(titles[i], fontsize=32)

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    if filename is not None:
        anim.save(filename, writer="ffmpeg")
    if show:
        plt.show()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
    if th.cuda.is_available():
        th.cuda.manual_seed(seed)
        th.cuda.manual_seed_all(seed)

def unscale_action(action, action_space: gym.spaces.Box):
    low = action_space.low
    high = action_space.high
    return low + (0.5 * (action + 1.0) * (high - low))

class Trainer:
    def __init__(self, env: gym.Env, tensorboard_log: str, seed = 0, num_steps = 3*10**6):
        self.env = env
        self.env.action_space.seed(seed)

        self.tensorboard_log = tensorboard_log
        self.writer = SummaryWriter(log_dir=tensorboard_log)

        self.num_steps = num_steps
    
    def learn(self, eval_env: gym.Env, eval_interval = 10**4, num_eval_episodes=5):
        return NotImplementedError
    
    def evaluate(self, eval_env: gym.Env, step: int, num_eval_episodes: int):
        return NotImplementedError
    
    def save(self, save_dir):
        return NotImplementedError
