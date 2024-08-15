import hydra
from omegaconf import DictConfig

from trainer_vp import Trainer

@hydra.main(config_path="../config/ocrl", config_name="trainer_ocq_vp")
def main(cfg: DictConfig):
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
