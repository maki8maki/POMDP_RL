import gymnasium as gym
import numpy as np
import os
from tqdm import tqdm

from slac.algo import SlacAlgorithm
from slac.utils import SlacObservation
from utils import Trainer, unscale_action, anim

class SLACTrainer(Trainer):
    """
    Trainer for SLAC.
    """

    def __init__(
        self,
        env: gym.Env,
        tensorboard_log: str,
        algo_kwargs : dict = {},
        device = 'cpu',
        seed = 0,
        num_steps = 3 * 10 ** 6,
        initial_collection_steps = 10 ** 4,
        initial_learning_steps = 10 ** 5,
    ):
        super().__init__(env, tensorboard_log, seed, num_steps)

        # Algorithm to learn.
        self.algo = SlacAlgorithm(env.observation_space.shape, env.action_space.shape, device, **algo_kwargs)

        # Observations for training and evaluation.
        num_sequences = self.algo.num_sequences
        self.ob = SlacObservation(env.observation_space.shape, env.action_space.shape, num_sequences)
        self.ob_test = SlacObservation(env.observation_space.shape, env.action_space.shape, num_sequences)

        # Other parameters.
        self.initial_collection_steps = initial_collection_steps
        self.initial_learning_steps = initial_learning_steps
        self.algo.learning_steps_sac = initial_collection_steps - 1

    def learn(self, eval_env: gym.Env, eval_interval = 10**4, num_eval_episodes = 5):
        # Initialize the environment.
        state, _ = self.env.reset()
        self.ob.reset_episode(state)
        self.algo.buffer.reset_episode(state)

        for step in tqdm(range(1, self.num_steps+1)):
            self.algo.step(self.env, self.ob, step<=self.initial_collection_steps)
            
            if step == self.initial_collection_steps:
                bar = tqdm((range(self.initial_learning_steps)), desc="Updating latent variable model.", leave=False)
                for _ in bar:
                    self.algo.update_latent(self.writer)
            elif step > self.initial_collection_steps:
                self.algo.update_latent(self.writer)
                self.algo.update_sac(self.writer)
            
            if step % eval_interval == 0:
                self.evaluate(eval_env, step, num_eval_episodes)
        self.writer.flush()
        self.writer.close()

    def evaluate(self, eval_env: gym.Env, step: int, num_eval_episodes):
        results = []

        for _ in range(num_eval_episodes):
            state, _ = eval_env.reset()
            self.ob_test.reset_episode(state)
            episode_return = 0.0
            done = False

            while not done:
                action = self.algo.exploit(self.ob_test)
                state, reward, terminated, truncated, _ = eval_env.step(unscale_action(action, eval_env.action_space))
                done = terminated or truncated
                self.ob_test.append(state, action)
                episode_return += reward

            results.append(episode_return)
        
        self.timesteps.append(step)
        self.episode_results.append(results)
        log_path = os.path.join(self.tensorboard_log, 'evaluations')
        np.savez(log_path, timesteps=self.timesteps, results=self.episode_results)

        # Log to TensorBoard.
        self.writer.add_scalar("eval/mean_reward", np.mean(results), step)

    def view(self):
        state, _ = self.env.reset()
        self.ob_test.reset_episode(state)
        frames = [self.env.render()]
        step = 0
        titles = [f'Step {step}']
        episode_return = 0.0
        done = False
        while not done:
            step += 1
            action = self.algo.exploit(self.ob_test)
            state, reward, terminated, truncated, _ = self.env.step(unscale_action(action, self.env.action_space))
            done = terminated or truncated
            self.ob_test.append(state, action)
            episode_return += reward
            frames.append(self.env.render())
            titles.append(f'Step {step}')
        print(episode_return)
        anim(frames=frames, titles=titles, filename='logs/slac.gif', show=False)
    
    def save(self, save_dir):
        self.algo.save_model(save_dir)

    def load(self, path):
        self.algo.load_model(path)
