import hydra
from omegaconf import DictConfig

from trainer_slate import Trainer

@hydra.main(config_path="../config/ocrl", config_name="trainer_vanilla_slate")
def main(cfg: DictConfig):
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
