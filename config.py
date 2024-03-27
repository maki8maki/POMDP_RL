import dacite
import dataclasses
import gymnasium as gym
import hydra
from omegaconf import OmegaConf
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from typing import Union

from utils import make_env, set_seed, Trainer

@dataclasses.dataclass
class Config:
    _env: dataclasses.InitVar[dict] = None
    env: gym.Env = dataclasses.field(default=None)
    _model: dataclasses.InitVar[dict] = None
    model: Union[BaseAlgorithm, Trainer] = dataclasses.field(default=None)
    seed: dataclasses.InitVar[int] = None
    learn_kwargs: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self, _env: dict, _model: dict, seed: int):
        self.env = make_env(**_env)
        self.model = hydra.utils.instantiate(
            _model,
            env=self.env,
            seed=seed,
            tensorboard_log=hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        )
        if isinstance(self.model, BaseAlgorithm):
            eval_env = Monitor(make_env(**_env))
            self.learn_kwargs['callback'] = EvalCallback(
                eval_env=eval_env,
                eval_freq=self.learn_kwargs['total_timesteps']/100,
                verbose=0
            )
        elif isinstance(self.model, Trainer):
            set_seed(seed)
            eval_env = make_env(**_env)
            self.learn_kwargs.update({
                'eval_env': eval_env,
                'eval_interval': self.model.num_steps / 100
            })
    
    @classmethod
    def convert(cls, _cfg: OmegaConf):
        cfg = dacite.from_dict(data_class=cls, data=OmegaConf.to_container(_cfg))
        return cfg
