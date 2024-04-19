import hydra
from omegaconf import DictConfig

from trainer_pretrain_token_vis import Trainer

@hydra.main(config_path="../config/ocrl", config_name="trainer_oc_pretrain_token-vis")
def main(cfg: DictConfig):
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
