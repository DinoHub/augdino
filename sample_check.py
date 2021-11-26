import hydra
from omegaconf import DictConfig

@hydra.main(config_path='.', config_name='sample_config')
def main(cfg: DictConfig) -> None:
    print(cfg)

if __name__ == '__main__':
    main()