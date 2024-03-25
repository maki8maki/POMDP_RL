import gymnasium as gym
import hydra
from matplotlib import animation
import matplotlib.pyplot as plt

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
