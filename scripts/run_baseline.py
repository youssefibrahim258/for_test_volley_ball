import warnings
warnings.filterwarnings("ignore")
import argparse
import yaml
from src.pipelines.B1.train_b1 import train_b1
from src.pipelines.B1.eval_b1 import eval_b1
from src.pipelines.B3.train_b3 import train_b3
from src.pipelines.B3.eval_b3 import eval_b3
from src.pipelines.B3_satge2.train_b3_stage2 import train_stage2
from src.pipelines.B3_satge2.val_b3_stage2 import eval_b3_stage2
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/B3_b.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if cfg["baseline"] == "B1":
        train_b1(cfg)
        eval_b1(cfg)
    
    elif cfg["baseline"] == "B3":
        train_b3(cfg)
        eval_b3(cfg)

    elif cfg["baseline"] == "B3_stage2":
        train_stage2(cfg)
        eval_b3_stage2(cfg)


    else:
        raise NotImplementedError(f"{cfg['baseline']} not implemented yet")


if __name__ == "__main__":

    main()


# python -m  scripts.run_baseline