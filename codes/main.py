import os
import time
import wandb
import traceback
import argparse

from run import C_HNTM_Runner

parser = argparse.ArgumentParser("NTM")
parser.add_argument("--data", type=str, default="20news", choices=["20news", "wiki103", "reuters"], help="数据集")
parser.add_argument("--num_topics", type=int, default=100)
parser.add_argument("--num_clusters", type=int, default=20)

parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--pre_num_epochs", type=int, default=50)
parser.add_argument("--num_epochs", type=int, default=100)
parser.add_argument("--hidden_dim", type=int, default=256)
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--topk_words", type=int, default=30, help="主题词个数，用于打印结果和初始化依赖矩阵")

parser.add_argument("--metric_log_interval", type=int, default=10)
parser.add_argument("--topic_log_interval", type=int, default=20)
parser.add_argument("--wandb", action="store_true", default=False)
parser.add_argument("--pretrain", action="store_true", default=False)

args = parser.parse_args()

# wandb configuration
if args.wandb:
    wandb.login()
    wandb.init(
        project = "c_hntm",
        name = time.strftime("%Y-%m-%d-%H-%M", time.localtime()),
        config=args
    )


def main():
    mode = "train"
    # mode = "load"
    if mode == "train":
        runner = C_HNTM_Runner(args, mode=mode)
        try:
            runner.train()
        except KeyboardInterrupt as e:
            print("KeyboardInterrupt... Aborted.")
        except Exception as e:
            print(traceback.print_exc())
    elif mode == "load":
        model_path = "../models/c_hntm/c_hntm_wiki103.pkl"
        runner = C_HNTM_Runner(args, mode=mode)
        runner.load(model_path)
        runner.show_hierachical_topic_results()


def test():
    runner = C_HNTM_Runner(args, mode="train")
    # root_topics = runner.get_root_topic_words(runner.model)
    # print("init root topics:")
    # for i,words in enumerate(root_topics):
    #     print("topic-{}: {}".format(i, words))
    try:
        runner.train()
    except KeyboardInterrupt as e:
        print("KeyboardInterrupt... Aborted.")
    except Exception as e:
        print(traceback.print_exc())    




if __name__ == "__main__":
    # main()
    test()