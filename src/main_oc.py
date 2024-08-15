import hydra
from omegaconf import DictConfig

from trainer_oc import Trainer

@hydra.main(config_path="../config/ocrl", config_name="trainer_ocq_goal")
def main(cfg: DictConfig):
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
