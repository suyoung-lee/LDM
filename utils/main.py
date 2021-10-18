"""
Main scripts to start experiments.
Takes a flag --env-type (see below for choices) and loads the parameters used in the paper.
"""
import argparse
import glob
import os
import warnings

import torch

# get configs
from config.gridworld import args_grid_rl2, args_grid_varibad, args_grid_ldm
from config.mujoco import args_mujoco_ant_dir_rl2, args_mujoco_ant_dir_varibad, args_mujoco_ant_dir_ldm
from config.mujoco import args_mujoco_ant_goal_rl2, args_mujoco_ant_goal_varibad, args_mujoco_ant_goal_ldm
from config.mujoco import args_mujoco_cheetah_vel_rl2, args_mujoco_cheetah_vel_varibad, args_mujoco_cheetah_vel_ldm

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--env-type', default='gridworld_varibad')
    args, rest_args = parser.parse_known_args()
    env = args.env_type

    # --- GridWorld ---
    if env == 'gridworld_varibad':
        args = args_grid_varibad.get_args(rest_args)
        from metalearner import MetaLearner
    elif env == 'gridworld_rl2':
        args = args_grid_rl2.get_args(rest_args)
        from metalearner import MetaLearner
    elif env == 'gridworld_ldm':
        from metalearner_ldm import MetaLearner
        args = args_grid_ldm.get_args(rest_args)
        args2 = args_grid_rl2.get_args(rest_args)

    # --- AntDir ---
    if env == 'mujoco_ant_dir_varibad':
        args = args_mujoco_ant_dir_varibad.get_args(rest_args)
        from metalearner import MetaLearner
    elif env == 'mujoco_ant_dir_rl2':
        args = args_mujoco_ant_dir_rl2.get_args(rest_args)
        from metalearner import MetaLearner
    elif env == 'mujoco_ant_dir_ldm':
        from metalearner_ldm import MetaLearner
        args = args_mujoco_ant_dir_ldm.get_args(rest_args)
        args2 = args_mujoco_ant_dir_rl2.get_args(rest_args)

    # --- AntGoal ---
    if env == 'mujoco_ant_goal_varibad':
        args = args_mujoco_ant_goal_varibad.get_args(rest_args)
        from metalearner import MetaLearner
    elif env == 'mujoco_ant_goal_rl2':
        args = args_mujoco_ant_goal_rl2.get_args(rest_args)
        from metalearner import MetaLearner
    elif env == 'mujoco_ant_goal_ldm':
        from metalearner_ldm import MetaLearner
        args = args_mujoco_ant_goal_ldm.get_args(rest_args)
        args2 = args_mujoco_ant_goal_rl2.get_args(rest_args)

    # --- CheetahVel ---
    if env == 'mujoco_cheetah_vel_varibad':
        args = args_mujoco_cheetah_vel_varibad.get_args(rest_args)
        from metalearner import MetaLearner
    elif env == 'mujoco_cheetah_vel_rl2':
        args = args_mujoco_cheetah_vel_rl2.get_args(rest_args)
        from metalearner import MetaLearner
    elif env == 'mujoco_cheetah_vel_ldm':
        from metalearner_ldm import MetaLearner
        args = args_mujoco_cheetah_vel_ldm.get_args(rest_args)
        args2 = args_mujoco_cheetah_vel_rl2.get_args(rest_args)

    # warning
    if args.deterministic_execution:
        print('Envoking deterministic code execution.')
        if torch.backends.cudnn.enabled:
            warnings.warn('Running with deterministic CUDNN.')
        if args.num_processes > 1:
            raise RuntimeError('If you want fully deterministic code, run it with num_processes=1.'
                               'Warning: This will slow things down and might break A2C if '
                               'policy_num_steps < env._max_episode_steps.')

    # start training
    if args.disable_varibad:
        # When the flag `disable_varibad` is activated, the file `learner.py` will be used instead of `metalearner.py`.
        # This is a stripped down version without encoder, decoder, stochastic latent variables, etc.
        learner = Learner(args)
    else:
        if env in ['gridworld_ldm', 'mujoco_ant_dir_ldm', 'mujoco_ant_goal_ldm', 'mujoco_cheetah_vel_ldm']:
            learner = MetaLearner(args, args2)
        else:
            learner = MetaLearner(args)
    learner.train()


if __name__ == '__main__':
    main()
