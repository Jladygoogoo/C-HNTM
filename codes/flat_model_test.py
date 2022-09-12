import os
import sys
import argparse
import torch
import wandb
from torch.utils import data
from gensim.models import Word2Vec

from flat_model_run import *


parser = argparse.ArgumentParser("NTM")
parser.add_argument("--model", type=str, choices=["gsm", "avitm", "etm"])
parser.add_argument("--data", type=str, choices=["20news", "reuters", "wiki103"])
parser.add_argument("--num_topics", type=int, default=100)
parser.add_argument("--num_epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--device", type=int, default=0)

parser.add_argument("--metric_log_interval", type=int, default=10)
parser.add_argument("--topic_log_interval", type=int, default=20)
parser.add_argument("--wandb", action="store_true", default=False)

args = parser.parse_args()


# wandb configuration
if args.wandb:
    wandb.login()
    wandb.init(
        project = "flat_model_test",
        name = time.strftime("%Y-%m-%d-%H-%M", time.localtime()),
        config=args
    )

def main():
    if args.model=="gsm":
        runner = NVDM_GSM_Runner(args)
    elif args.model=="avitm":
        runner = AVITM_Runner(args)
    elif args.model=="etm":
        runner = ETM_Runner(args)
    try:
        runner.train()
    except KeyboardInterrupt as e:
        print("KeyboardInterrupt...Terminated.")
        sys.exit(0)


if __name__=="__main__":
    main()
