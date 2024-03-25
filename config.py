import dacite
import dataclasses
import gymnasium as gym
import hydra
from omegaconf import OmegaConf
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from utils import make_env

@dataclasses.dataclass
class Config:
    _env: dataclasses.InitVar[dict] = None
    env: gym.Env = dataclasses.field(default=None)
    _model: dataclasses.InitVar[dict] = None
    model: BaseAlgorithm = dataclasses.field(default=None)
    seed: dataclasses.InitVar[int] = None
    learn_kwargs: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self, _env, _model, seed):
        self.env = make_env(**_env)
        self.model: BaseAlgorithm = hydra.utils.instantiate(
            _model,
            env=self.env,
            seed=seed,
            tensorboard_log=hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        )
        eval_env = Monitor(make_env(**_env))
        self.learn_kwargs['callback'] = EvalCallback(
            eval_env=eval_env,
            best_model_save_path=self.model.tensorboard_log,
            eval_freq=self.learn_kwargs['total_timesteps']/100,
            verbose=0
        )
    
    @classmethod
    def convert(cls, _cfg: OmegaConf):
        cfg = dacite.from_dict(data_class=cls, data=OmegaConf.to_container(_cfg))
        return cfg
