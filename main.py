import torch
import argparse
from argparse import ArgumentParser
from src.train import train
from utils.utils import strtobool
import os
import local_config
import wandb
from functools import partial 


def main(hyperparams):
    # run = wandb.init()
    print(hyperparams)
    epoch = hyperparams.epoch
    lr = hyperparams.lr
    batch_size = hyperparams.batch_size
    model = hyperparams.model

    last_layer = hyperparams.last_layer
    device = hyperparams.device
    strategy = hyperparams.finetune_strategy
    path = hyperparams.path
    seed = hyperparams.seed
    lora = hyperparams.lora
    lora_r = hyperparams.lora_r
    # lora_r = wandb.config.lora_r 
    l1_reg = hyperparams.l1_reg
    l1_lambda = hyperparams.l1_lambda

    train(model,
          epoch,
          batch_size,
          lr,
          device,
          last_layer,
          strategy,
          path,
          seed,
          lora,
          lora_r,
          l1_reg,
          l1_lambda,
    )


if __name__ == "__main__":
    # todo setup wandb key and username
    os.environ['WANDB_API_KEY'] = local_config.WANDB_KEY
    os.environ['WANDB_ENTITY'] = local_config.WANDB_USERNAME
    os.environ["WANDB_MODE"] = "offline"
    # wandb.login()

    Parser = ArgumentParser()
    Parser.register('type', 'bool', strtobool)

    # training hyperparameters
    Parser.add_argument("--epoch", type=int, default=200)
    Parser.add_argument('--lr', type=float, default=5e-5)
    Parser.add_argument('--batch_size', type=int, default=64)
    Parser.add_argument('--last_layer', type='bool', default=False, help='Only optimize the last layer')
    Parser.add_argument('--lora', type='bool', default=True, help='whether use lora')
    Parser.add_argument('--lora_r', type=int, default=10, help='hyperparameter for lora rank')
    # Parser.add_argument("--lora_r", nargs="+", help='hyperparameter for lora rank')
    Parser.add_argument("--device", default="cuda", type=str)
    Parser.add_argument("--model", default="SoftmaxBERT", type=str)
    Parser.add_argument("--finetune_strategy", type=str)
    Parser.add_argument("--seed", type=int, default=42)
    Parser.add_argument("--l1_lambda", type=int, default=0.01)

    Parser.add_argument("--l1_reg", type='bool', default=False, help='whether to sparsify the lora matrix using l1 reg')



    # model checkpoint
    Parser.add_argument("--path",default="None", type=str)
    Parser.add_argument("--frequency", type=int, help="how often to save the model", default=1000)

    # todo get models from command line i.e., backbone + head
    # partially done, getting the whole model directly from the cli

    hyperparams = Parser.parse_args()

    # sweep_config = {
    #     'method': "grid",
    #     'parameters':{
    #         'lora_r': {"values": [40,50,60,70,80,90,100]},
    #     }
    # }
    
    # sweep_id = wandb.sweep(sweep=sweep_config, project="lora_r hyperparameter")
    # train = partial(main, hyperparams)
    # wandb.agent(sweep_id, function=train)
    main(hyperparams)
    