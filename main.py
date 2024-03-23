import hydra
from omegaconf import OmegaConf
import os

from config import Config

@hydra.main(config_path='config/', config_name='config', version_base=None)
def main(_cfg: OmegaConf):
    cfg = Config.convert(_cfg)
    cfg.model.learn(**cfg.learn_kwargs)
    cfg.model.save(os.path.join(cfg.model.tensorboard_log, 'model'))

if __name__ == '__main__':
    main()
