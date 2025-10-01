import json
from omegaconf import OmegaConf, DictConfig
import hydra
from pathlib import Path
from train_helper import train

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    print(OmegaConf.to_yaml(cfg))
    results = train(cfg_dict)

    cfg_dict["results"] = results

    out = Path("results.jsonl")
    with out.open("a") as f:
        f.write(json.dumps(cfg_dict) + "\n")


if __name__ == "__main__":
    main()