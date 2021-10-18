# Latent Dynamics Mixture

PyTorch implementation of "Improving Generalization in Meta-RL with Imaginary Tasks from Latent Dynamics Mixture", NeurIPS2021.

# Requirements

Our code is based on the reference implementation of [variBAD](https://github.com/lmzintgraf/varibad).

Refer to the requirements in https://github.com/lmzintgraf/varibad. 

You don't need MuJoCo license to run the gridworld experiment.

# How to Run

`python main.py --env-type gridworld_ldm`

`python main.py --env-type mujoco_ant_dir_ldm`

`python main.py --env-type mujoco_ant_goal_ldm`

`python main.py --env-type mujoco_cheetah_vel_ldm`



If you want to run rl2 and varibad,

`python main.py --env-type (envname)_rl2`
  
`python main.py --env-type (envname)_varibad`

Evaluation results will be stored in the `logs` folder
If you want to change the configurations of LDM, refer to configurations in the `config` folder
The major part of the algorithm is in `metalearner_ldm.py`.










