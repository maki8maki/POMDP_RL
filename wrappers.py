import cv2
import gymnasium as gym
from gymnasium.wrappers import FrameStack, GrayScaleObservation
import numpy as np
from typing import Any

class ImageObservation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, width: int = 100, height: int = 100):
        super().__init__(env)

        self.width = width
        self.height = height
        self.env.reset()
        image = self.observation(None)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=image.shape, dtype=np.uint8)
    
    def observation(self, observation: Any) -> np.ndarray:
        return cv2.resize(self.env.render(), (self.width, self.height))

class MultiGrayImageObservation(gym.Wrapper):
    def __init__(self, env: gym.Env, width: int = 100, height: int = 100, num_stack: int = 4):
        super().__init__(env)
        self.env = ImageObservation(env=self.env, width=width, height=height)
        self.env = GrayScaleObservation(self.env)
        self.env = FrameStack(self.env, num_stack=num_stack)
