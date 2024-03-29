import gymnasium as gym
import numpy as np
import os
from tqdm import tqdm

from dreamer.buffer import ReplayBuffer
from dreamer.core import Agent, Dreamer
from utils import anim, unscale_action, Trainer

class DreamerTrainer(Trainer):
    def __init__(
            self,
            env: gym.Env,
            tensorboard_log: str,
            seed = 0,
            num_steps = 3*10**6,
            buffer_capacity = 200000,
            dreamer_config: dict = {},
            initial_collection_episodes = 5,
            verbose=False
        ):
        super().__init__(env, tensorboard_log, seed, num_steps)
        self.dreamer = Dreamer(self.env.observation_space.shape, self.env.action_space.shape[0], **dreamer_config)
        self.buffer = ReplayBuffer(buffer_capacity, self.env.observation_space.shape, self.env.action_space.shape[0])

        self.initial_collection_episodes = initial_collection_episodes

        self.verbose = verbose
        self.make_aliases()
    
    def make_aliases(self):
        self.encoder = self.dreamer.encoder
        self.rssm = self.dreamer.rssm
        self.obs_model = self.dreamer.obs_model
        self.reward_model = self.dreamer.reward_model
        self.value_model = self.dreamer.value_model
        self.action_model = self.dreamer.action_model
    
    def learn(self, eval_env: gym.Env, eval_interval=10**4, num_eval_episodes=5, collect_interval=100, action_noise_var=0.3,
              batch_size=50, chunk_length=50, free_nats=3, clip_grad_norm=100, imagination_horizon=15, gamma=0.9, lambda_=0.95):
        policy = Agent(self.encoder, self.rssm, self.action_model)
        obs, _ = self.env.reset()
        total_reward = 0.0
        total_update_step = self.initial_collection_episodes * collect_interval
        for step in tqdm(range(1, self.num_steps+1)):
            is_collection = (step <= self.initial_collection_episodes * self.env.get_wrapper_attr('_max_episode_steps'))
            if is_collection:
                action = self.env.action_space.sample()
            else:
                action = policy(obs)
                action += np.random.normal(0, np.sqrt(action_noise_var), self.env.action_space.shape[0])
            next_obs, reward, terminated, truncated, _ = self.env.step(unscale_action(action, self.env.action_space))
            done = terminated or truncated
            self.buffer.push(obs, action, reward, done)
            total_reward += reward
            if done:
                obs, _ = self.env.reset()
                self.writer.add_scalar('train/total_reward', total_reward, step)
                total_reward = 0.0
                if not is_collection:
                    for _ in tqdm(range(collect_interval), leave=False, desc='Updating...'):
                        observations, actions, rewards, _ = self.buffer.sample(batch_size, chunk_length)
                        losses = self.dreamer.update(observations, actions, rewards, free_nats, clip_grad_norm, imagination_horizon, gamma, lambda_)

                        total_update_step += 1
                        for key, value in losses.items():
                            self.writer.add_scalar(f'train/{key}', value.item(), total_update_step)
                    policy = Agent(self.encoder, self.rssm, self.action_model)
            else:
                obs = next_obs
            if step % eval_interval == 0:
                self.evaluate(eval_env, step, num_eval_episodes)
        self.writer.flush()
        self.writer.close()
    
    def evaluate(self, eval_env: gym.Env, step: int, num_eval_episodes=5):
        self.dreamer.eval()
        policy = Agent(self.encoder, self.rssm, self.action_model)
        results = []
        for _ in range(num_eval_episodes):
            obs, _ = eval_env.reset()
            done = False
            total_reward = 0.0
            while not done:
                action = policy(obs, training=False)
                next_obs, reward, terminated, truncated, _ = eval_env.step(unscale_action(action, eval_env.action_space))
                done = terminated or truncated
                obs = next_obs
                total_reward += reward
            results.append(total_reward)
        
        self.timesteps.append(step)
        self.episode_results.append(results)
        log_path = os.path.join(self.tensorboard_log, 'evaluations')
        np.savez(log_path, timesteps=self.timesteps, results=self.episode_results)

        self.dreamer.train()
        self.writer.add_scalar("eval/mean_reward", np.mean(results), step)
    
    def view(self):
        self.dreamer.eval()
        policy = Agent(self.encoder, self.rssm, self.action_model)
        total_reward = 0.0
        obs, _ = self.env.reset()
        frames = [self.env.render()]
        step = 0
        titles = [f'Step {step}']
        done = False
        while not done:
            step += 1
            action = policy(obs, training=False)
            next_obs, reward, terminated, truncated, _ = self.env.step(unscale_action(action, self.env.action_space))
            done = terminated or truncated
            obs = next_obs
            total_reward += reward
            frames.append(self.env.render())
            titles.append(f'Step {step}')
        print(total_reward)
        anim(frames=frames, titles=titles, filename='logs/dreamer.gif', show=False)
    
    def save(self, log_dir='logs', name='params.pth'):
        os.makedirs(log_dir, exist_ok=True)
        self.dreamer.save(os.path.join(log_dir, name))
    
    def close(self):
        self.env.close()
    
    def load(self, path):
        self.dreamer.load(path)
