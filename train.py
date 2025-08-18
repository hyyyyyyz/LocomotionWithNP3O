import numpy as np
import os
from datetime import datetime
from configs.go2_constraint_him import Go2ConstraintHimRoughCfg, Go2ConstraintHimRoughCfgPPO
from configs.go2_constraint_trans_p1 import Go2ConstraintTransP1RoughCfg, Go2ConstraintTransP1RoughCfgPPO
from configs.go2_constraint_trans_p2 import Go2ConstraintTransP2RoughCfg, Go2ConstraintTransP2RoughCfgPPO
from configs.tinymal_constraint_him import TinymalConstraintHimRoughCfg, TinymalConstraintHimRoughCfgPPO
import isaacgym
from utils.helpers import get_args
from envs import LeggedRobot
from utils.task_registry import task_registry
# python train.py --task=go2N3poHim
# python train.py --task=Tinymal

def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    task_registry.register("go2N3poHim",LeggedRobot,Go2ConstraintHimRoughCfg(),Go2ConstraintHimRoughCfgPPO())
    task_registry.register("Tinymal",LeggedRobot,TinymalConstraintHimRoughCfg(),TinymalConstraintHimRoughCfgPPO())
    task_registry.register("go2N3poTransP1",LeggedRobot,Go2ConstraintTransP1RoughCfg(),Go2ConstraintTransP1RoughCfgPPO())
    task_registry.register("go2N3poTransP2",LeggedRobot,Go2ConstraintTransP2RoughCfg(),Go2ConstraintTransP2RoughCfgPPO())
    args = get_args()
    # args.task='Tinymal'
    train(args)
