import itertools
import math
import random

import gym
gym.logger.set_level(40)
import matplotlib.pyplot as plt
import numpy as np
import torch
from gym import spaces
from matplotlib.patches import Rectangle

from utils import helpers as utl

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class GridNavi(gym.Env):
    def __init__(self,
                 num_cells=5, num_steps=15,
                 oracle=False,  # oracle gets the true goal as input
                 belief_oracle=False,  # belief oracle gets the true belief as input
                 give_goal_hint_at_request=False,  # here the policy can *ask* for a hint (might come with a price)
                 ):
        super(GridNavi, self).__init__()

        self.seed()

        self.oracle = oracle
        self.belief_oracle = belief_oracle
        assert not (self.oracle and self.belief_oracle)

        self.enable_goal_hint = give_goal_hint_at_request

        self.num_cells = num_cells
        self.num_states = num_cells ** 2

        self._max_episode_steps = num_steps
        self.step_count = 0

        if self.belief_oracle:
            self.observation_space = spaces.Box(low=0, high=self.num_cells - 1, shape=(2 + self.num_cells ** 2,),
                                                dtype=np.float32)
        elif oracle or self.enable_goal_hint:
            self.observation_space = spaces.Box(low=0, high=self.num_cells - 1, shape=(2 + 2,), dtype=np.float32)
        else:
            self.observation_space = spaces.Box(low=0, high=self.num_cells - 1, shape=(2,), dtype=np.float32)

        if self.enable_goal_hint:
            self.action_space = spaces.Discrete(6)  # noop, up, right, down, left,  get hint
        else:
            self.action_space = spaces.Discrete(5)  # noop, up, right, down, left

        # possible starting states
        self.starting_states = [(0.0, 0.0)]  # , (self.num_cells-1, 0)]#,
        # (0, self.num_cells-1), (self.num_cells-1, self.num_cells-1)]

        # goals can be anywhere except on possible starting states
        self.possible_goals = list(itertools.product(range(num_cells), repeat=2))
        for s in self.starting_states:
            self.possible_goals.remove(s)
        self.possible_goals.remove((0, 1))
        self.possible_goals.remove((1, 1))
        self.possible_goals.remove((1, 0))

        #check interpolation 1 behavior #seed 89
        LDM = True
        if LDM:
            #self.possible_goals.remove((2, 0))
            #self.possible_goals.remove((2, 1))
            #self.possible_goals.remove((2, 2))
            #self.possible_goals.remove((1, 2))
            #self.possible_goals.remove((0, 2))

            self.possible_goals.remove((3, 0))
            self.possible_goals.remove((3, 1))
            self.possible_goals.remove((3, 2))
            self.possible_goals.remove((3, 3))
            self.possible_goals.remove((2, 3))
            self.possible_goals.remove((1, 3))
            self.possible_goals.remove((0, 3))

            self.possible_goals.remove((4, 0))
            self.possible_goals.remove((4, 1))
            self.possible_goals.remove((4, 2))
            self.possible_goals.remove((4, 3))
            self.possible_goals.remove((4, 4))
            self.possible_goals.remove((3, 4))
            self.possible_goals.remove((2, 4))
            self.possible_goals.remove((1, 4))
            self.possible_goals.remove((0, 4))

            self.possible_goals.remove((5, 0))
            self.possible_goals.remove((5, 1))
            self.possible_goals.remove((5, 2))
            self.possible_goals.remove((5, 3))
            self.possible_goals.remove((5, 4))
            self.possible_goals.remove((5, 5))
            self.possible_goals.remove((4, 5))
            self.possible_goals.remove((3, 5))
            self.possible_goals.remove((2, 5))
            self.possible_goals.remove((1, 5))
            self.possible_goals.remove((0, 5))

            #self.possible_goals.remove((6, 0))
            #self.possible_goals.remove((6, 1))
            #self.possible_goals.remove((6, 2))
            #self.possible_goals.remove((6, 3))
            #self.possible_goals.remove((6, 4))
            #self.possible_goals.remove((6, 5))
            #self.possible_goals.remove((6, 6))
            #self.possible_goals.remove((5, 6))
            #self.possible_goals.remove((4, 6))
            #self.possible_goals.remove((3, 6))
            #self.possible_goals.remove((2, 6))
            #self.possible_goals.remove((1, 6))
            #self.possible_goals.remove((0, 6))


        self.task_dim = 2
        self.num_tasks = self.num_states

        # reset the environment state
        self._env_state = np.array(random.choice(self.starting_states))
        # reset the goal
        self._goal = self.reset_task()
        # reset the belief
        if self.belief_oracle:
            self._belief_state = self._reset_belief()

    def get_task(self):
        return self._goal

    def get_possible_goals(self):
        return self.possible_goals

    def reset_task(self, task=None):
        if task is None:
            self._goal = np.array(random.choice(self.possible_goals))
        else:
            self._goal = np.array(task)
        if self.belief_oracle:
            self._reset_belief()
        return self._goal

    def _reset_belief(self):
        self._belief_state = np.zeros((self.num_cells ** 2))
        for pg in self.possible_goals:
            idx = self.task_to_id(np.array(pg))
            self._belief_state[idx] = 1.0 / len(self.possible_goals)
        return self._belief_state

    def update_belief(self, state, action):

        on_goal = state[0] == self._goal[0] and state[1] == self._goal[1]

        # hint
        if action == 5 or on_goal:
            possible_goals = self.possible_goals.copy()
            possible_goals.remove(tuple(self._goal))
            wrong_hint = possible_goals[random.choice(range(len(possible_goals)))]
            self._belief_state *= 0
            self._belief_state[self.task_to_id(self._goal)] = 0.5
            self._belief_state[self.task_to_id(wrong_hint)] = 0.5
        else:
            self._belief_state[self.task_to_id(state)] = 0
            self._belief_state = np.ceil(self._belief_state)
            self._belief_state /= sum(self._belief_state)

        return self._belief_state

    def reset(self):

        self.step_count = 0

        # reset agent to starting position
        self._env_state = np.array(random.choice(self.starting_states))

        # get initial observation, augmented with hint/belief
        if self.oracle:
            state = np.concatenate((self._env_state.copy(), self._goal.copy()))
        elif self.belief_oracle:
            state = np.concatenate((self._env_state.copy(), self._belief_state.copy()))
        elif self.enable_goal_hint:
            state = np.concatenate((self._env_state, np.zeros(2)))
        else:
            state = self._env_state.copy()

        return state

    def state_transition(self, action):
        """
        Moving the agent between states
        """

        if action == 1:  # up
            self._env_state[1] = min([self._env_state[1] + 1, self.num_cells - 1])
        elif action == 2:  # right
            self._env_state[0] = min([self._env_state[0] + 1, self.num_cells - 1])
        elif action == 3:  # down
            self._env_state[1] = max([self._env_state[1] - 1, 0])
        elif action == 4:  # left
            self._env_state[0] = max([self._env_state[0] - 1, 0])

        return self._env_state

    def step(self, action):

        if isinstance(action, np.ndarray) and action.ndim == 1:
            action = action[0]
        assert self.action_space.contains(action)

        done = False

        # perform state transition
        state = self.state_transition(action)

        # check if maximum step limit is reached
        self.step_count += 1
        if self.step_count >= self._max_episode_steps:
            done = True

        # compute reward
        if self._env_state[0] == self._goal[0] and self._env_state[1] == self._goal[1]:
            reward = 1.0
        elif self.enable_goal_hint and action == 5:
            reward = -1
        else:
            reward = -0.1

        # give hint / belief
        if self.belief_oracle:
            self.update_belief(self._env_state, action)
            state = np.concatenate((self._env_state.copy(), self._belief_state.copy()))
        elif self.oracle or self.enable_goal_hint:
            state = np.concatenate((state.copy(), np.zeros(2)))
            if self.oracle:
                state[-2:] = self._goal.copy()
            elif action == 5 and self.enable_goal_hint:
                state[-2:] = self._goal.copy()

        return state, reward, done, {'task': self.get_task()}

    def task_to_id(self, goals):
        mat = torch.arange(0, self.num_cells ** 2).long().reshape((self.num_cells, self.num_cells))
        if isinstance(goals, list) or isinstance(goals, tuple):
            goals = np.array(goals)
        if isinstance(goals, np.ndarray):
            goals = torch.from_numpy(goals)
        goals = goals.long()

        if goals.dim() == 1:
            goals = goals.unsqueeze(0)

        goal_shape = goals.shape
        if len(goal_shape) > 2:
            goals = goals.reshape(-1, goals.shape[-1])

        classes = mat[goals[:, 0], goals[:, 1]]
        classes = classes.reshape(goal_shape[:-1])

        return classes

    def id_to_task(self, classes):
        mat = torch.arange(0, self.num_cells ** 2).long().reshape((self.num_cells, self.num_cells)).numpy()
        goals = np.zeros((len(classes), 2))
        classes = classes.numpy()
        for i in range(len(classes)):
            pos = np.where(classes[i] == mat)
            goals[i, 0] = float(pos[0][0])
            goals[i, 1] = float(pos[1][0])
        goals = torch.from_numpy(goals).to(device).float()
        return goals

    def id_to_task_(self, classes):
        mat = torch.arange(0, self.num_cells ** 2).long().reshape((self.num_cells, self.num_cells)).numpy()
        goals = np.zeros((len(classes), 3))
        classes = classes.numpy()
        for i in range(len(classes)):
            pos = np.where(classes[i] == mat)
            goals[i, 0] = float(pos[0][0])
            goals[i, 1] = float(pos[1][0])
        return goals


    def goal_to_onehot_id(self, pos):
        cl = self.task_to_id(pos)
        if cl.dim() == 1:
            cl = cl.view(-1, 1)
        nb_digits = self.num_cells ** 2
        # One hot encoding buffer that you create out of the loop and just keep reusing
        y_onehot = torch.FloatTensor(pos.shape[0], nb_digits).to(device)
        # In your for loop
        y_onehot.zero_()
        y_onehot.scatter_(1, cl, 1)
        return y_onehot

    def onehot_id_to_goal(self, pos):
        if isinstance(pos, list):
            pos = [self.id_to_task(p.argmax(dim=1)) for p in pos]
        else:
            pos = self.id_to_task(pos.argmax(dim=1))
        return pos

    @staticmethod
    def visualise_behaviour(env,
                            args,
                            policy,
                            iter_idx,
                            encoder=None,
                            reward_decoder=None,
                            image_folder=None,
                            task_num = 0,
                            trial_num = None,
                            encoder_vae = None,
                            args_pol=None,
                            **kwargs
                            ):
        """
        Visualises the behaviour of the policy, together with the latent state and belief.
        The environment passed to this method should be a SubProcVec or DummyVecEnv, not the raw env!
        """

        num_episodes = args.max_rollouts_per_task
        unwrapped_env = env.venv.unwrapped.envs[0]

        # --- initialise things we want to keep track of ---

        episode_all_obs = [[] for _ in range(num_episodes)]
        episode_prev_obs = [[] for _ in range(num_episodes)]
        episode_next_obs = [[] for _ in range(num_episodes)]
        episode_actions = [[] for _ in range(num_episodes)]
        episode_rewards = [[] for _ in range(num_episodes)]

        episode_returns = []
        episode_lengths = []

        episode_goals = []
        if getattr(unwrapped_env, 'belief_oracle', False):
            episode_beliefs = [[] for _ in range(num_episodes)]
        else:
            episode_beliefs = None

        if encoder is not None:
            # keep track of latent spaces
            episode_latent_samples = [[] for _ in range(num_episodes)]
            episode_latent_means = [[] for _ in range(num_episodes)]
            episode_latent_logvars = [[] for _ in range(num_episodes)]
        else:
            episode_latent_samples = episode_latent_means = episode_latent_logvars = None

        if encoder_vae is not None:
            # keep track of latent spaces
            episode_latent_samples_vae = [[] for _ in range(num_episodes)]
            episode_latent_means_vae = [[] for _ in range(num_episodes)]
            episode_latent_logvars_vae = [[] for _ in range(num_episodes)]
        else:
            episode_latent_samples_vae = episode_latent_means_vae = episode_latent_logvars_vae = None

        curr_latent_sample = curr_latent_mean = curr_latent_logvar = None

        # --- roll out policy ---
        eval_possible_goals = \
            [(0, 2), (0, 3), (0, 4), (0, 5), (0, 6),
             (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
             (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6),
             (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6),
             (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6),
             (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6),
             (6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6)]


        (obs_raw, obs_normalised) = env.reset()
        env.reset_task(task=eval_possible_goals[task_num])

        obs_raw = obs_raw.float().reshape((1, -1)).to(device)
        obs_normalised = obs_normalised.float().reshape((1, -1)).to(device)
        start_obs_raw = obs_raw.clone()

        for episode_idx in range(args.max_rollouts_per_task):

            curr_goal = env.get_task()
            curr_rollout_rew = []
            curr_rollout_goal = []

            if encoder is not None:

                if episode_idx == 0:
                    # reset to prior
                    curr_latent_sample, curr_latent_mean, curr_latent_logvar, hidden_state = encoder.prior(1)
                    curr_latent_sample = curr_latent_sample[0].to(device)
                    curr_latent_mean = curr_latent_mean[0].to(device)
                    curr_latent_logvar = curr_latent_logvar[0].to(device)

                episode_latent_samples[episode_idx].append(curr_latent_sample[0].clone())
                episode_latent_means[episode_idx].append(curr_latent_mean[0].clone())
                episode_latent_logvars[episode_idx].append(curr_latent_logvar[0].clone())

            if encoder_vae is not None:

                if episode_idx == 0:
                    # reset to prior
                    curr_latent_sample_vae, curr_latent_mean_vae, curr_latent_logvar_vae, hidden_state_vae = encoder_vae.prior(1)
                    curr_latent_sample_vae = curr_latent_sample_vae[0].to(device)
                    curr_latent_mean_vae = curr_latent_mean_vae[0].to(device)
                    curr_latent_logvar_vae = curr_latent_logvar_vae[0].to(device)

                episode_latent_samples_vae[episode_idx].append(curr_latent_sample_vae[0].clone())
                episode_latent_means_vae[episode_idx].append(curr_latent_mean_vae[0].clone())
                episode_latent_logvars_vae[episode_idx].append(curr_latent_logvar_vae[0].clone())

            episode_all_obs[episode_idx].append(start_obs_raw.clone())
            if getattr(unwrapped_env, 'belief_oracle', False):
                episode_beliefs[episode_idx].append(unwrapped_env.unwrapped._belief_state.copy())

            for step_idx in range(1, env._max_episode_steps + 1):

                if step_idx == 1:
                    episode_prev_obs[episode_idx].append(start_obs_raw.clone())
                else:
                    episode_prev_obs[episode_idx].append(obs_raw.clone())

                # act
                _, action, _ = utl.select_action(args=args,
                                                 policy=policy,
                                                 obs=obs_normalised if args.norm_obs_for_policy else obs_raw,
                                                 deterministic=True, ###This may be changed when unseen environment appears
                                                 #deterministic = False, #207 to check adaptation performance
                                                 latent_sample=curr_latent_sample, latent_mean=curr_latent_mean,
                                                 latent_logvar=curr_latent_logvar)

                # observe reward and next obs
                (obs_raw, obs_normalised), (rew_raw, rew_normalised), done, infos = utl.env_step(env, action)
                obs_raw = obs_raw.reshape((1, -1)).to(device)
                obs_normalised = obs_normalised.reshape((1, -1)).to(device)

                if encoder is not None:
                    # update task embedding
                    curr_latent_sample, curr_latent_mean, curr_latent_logvar, hidden_state = encoder(
                        action.float().to(device),
                        obs_raw,
                        rew_raw.reshape((1, 1)).float().to(device),
                        hidden_state,
                        return_prior=False)

                    episode_latent_samples[episode_idx].append(curr_latent_sample[0].clone())
                    episode_latent_means[episode_idx].append(curr_latent_mean[0].clone())
                    episode_latent_logvars[episode_idx].append(curr_latent_logvar[0].clone())

                if encoder_vae is not None:
                    # update task embedding
                    curr_latent_sample_vae, curr_latent_mean_vae, curr_latent_logvar_vae, hidden_state_vae = encoder_vae(
                        action.float().to(device),
                        obs_raw,
                        rew_raw.reshape((1, 1)).float().to(device),
                        hidden_state_vae,
                        return_prior=False)

                    episode_latent_samples_vae[episode_idx].append(curr_latent_sample_vae[0].clone())
                    episode_latent_means_vae[episode_idx].append(curr_latent_mean_vae[0].clone())
                    episode_latent_logvars_vae[episode_idx].append(curr_latent_logvar_vae[0].clone())

                episode_all_obs[episode_idx].append(obs_raw.clone())
                episode_next_obs[episode_idx].append(obs_raw.clone())
                episode_rewards[episode_idx].append(rew_raw.clone())
                episode_actions[episode_idx].append(action.clone())

                curr_rollout_rew.append(rew_raw.clone())
                curr_rollout_goal.append(env.get_task().copy())

                if getattr(unwrapped_env, 'belief_oracle', False):
                    episode_beliefs[episode_idx].append(unwrapped_env.unwrapped._belief_state.copy())

                if infos[0]['done_mdp'] and not done:
                    start_obs_raw = infos[0]['start_state']
                    start_obs_raw = torch.from_numpy(start_obs_raw).float().reshape((1, -1)).to(device)
                    break

            episode_returns.append(sum(curr_rollout_rew))
            episode_lengths.append(step_idx)
            episode_goals.append(curr_goal)

        # clean up

        if encoder is not None:
            episode_latent_means = [torch.stack(e) for e in episode_latent_means]
            episode_latent_logvars = [torch.stack(e) for e in episode_latent_logvars]

        if encoder_vae is not None:
            episode_latent_means_vae = [torch.stack(e) for e in episode_latent_means_vae]
            episode_latent_logvars_vae = [torch.stack(e) for e in episode_latent_logvars_vae]

        episode_prev_obs = [torch.cat(e) for e in episode_prev_obs]
        episode_next_obs = [torch.cat(e) for e in episode_next_obs]
        episode_actions = [torch.cat(e) for e in episode_actions]
        episode_rewards = [torch.cat(e) for e in episode_rewards]

        # plot behaviour & visualise belief in env

        if encoder_vae is not None:
            rew_pred_means, rew_pred_vars = plot_bb(env, args, episode_all_obs, episode_goals, reward_decoder,
                                                    episode_latent_means_vae, episode_latent_logvars_vae,
                                                    image_folder, iter_idx, episode_beliefs, task_num,
                                                    episode_prev_obs, episode_actions, episode_rewards)
        else:
            rew_pred_means, rew_pred_vars = plot_bb(env, args, episode_all_obs, episode_goals, reward_decoder,
                                                    episode_latent_means, episode_latent_logvars,
                                                    image_folder, iter_idx, episode_beliefs, task_num,
                                                    episode_prev_obs, episode_actions, episode_rewards)

        if reward_decoder:
            plot_rew_reconstruction(env, rew_pred_means, rew_pred_vars, image_folder, iter_idx, task_num)


        if encoder_vae is not None:
            return episode_latent_means_vae, episode_latent_logvars_vae, \
                   episode_prev_obs, episode_next_obs, episode_actions, episode_rewards, \
                   episode_returns
        else:
            return episode_latent_means, episode_latent_logvars, \
                   episode_prev_obs, episode_next_obs, episode_actions, episode_rewards, \
                   episode_returns


def plot_rew_reconstruction(env,
                            rew_pred_means,
                            rew_pred_vars,
                            image_folder,
                            iter_idx,
                            task_num,
                            ):
    """
    Note that env might need to be a wrapped env!
    """

    num_rollouts = len(rew_pred_means)

    #plt.figure(figsize=(18, 5))
    plt.figure(figsize=(12, 5))

    #plt.subplot(1, 3, 1)
    plt.subplot(1, 2, 1)
    test_rew_mus = torch.cat(rew_pred_means).cpu().detach().numpy()
    plt.plot(range(test_rew_mus.shape[0]), test_rew_mus, '.-', alpha=0.5)
    plt.plot(range(test_rew_mus.shape[0]), test_rew_mus.mean(axis=1), 'k.-')
    for tj in np.cumsum([0, *[env._max_episode_steps for _ in range(num_rollouts)]]):
        span = test_rew_mus.max() - test_rew_mus.min()
        plt.plot([tj + 0.5, tj + 0.5], [test_rew_mus.min() - 0.05 * span, test_rew_mus.max() + 0.05 * span], 'k--',
                 alpha=0.5)
    plt.title('output - mean')


    #plt.subplot(1, 3, 2)
    plt.subplot(1, 2, 2)
    test_rew_vars = torch.cat(rew_pred_vars).cpu().detach().numpy()
    plt.plot(range(test_rew_vars.shape[0]), test_rew_vars, '.-', alpha=0.5)
    plt.plot(range(test_rew_vars.shape[0]), test_rew_vars.mean(axis=1), 'k.-')
    for tj in np.cumsum([0, *[env._max_episode_steps for _ in range(num_rollouts)]]):
        span = test_rew_vars.max() - test_rew_vars.min()
        plt.plot([tj + 0.5, tj + 0.5], [test_rew_vars.min() - 0.05 * span, test_rew_vars.max() + 0.05 * span],
                 'k--', alpha=0.5)
    plt.title('output - variance')

    """
    plt.subplot(1, 3, 3)
    rew_pred_entropy = -(test_rew_vars * np.log(test_rew_vars)).sum(axis=1)
    plt.plot(range(len(test_rew_vars)), rew_pred_entropy, 'r.-')
    for tj in np.cumsum([0, *[env._max_episode_steps for _ in range(num_rollouts)]]):
        span = rew_pred_entropy.max() - rew_pred_entropy.min()
        plt.plot([tj + 0.5, tj + 0.5], [rew_pred_entropy.min() - 0.05 * span, rew_pred_entropy.max() + 0.05 * span],
                 'k--', alpha=0.5)
    plt.title('Reward prediction entropy')
    """

    plt.tight_layout()
    if image_folder is not None:
        behaviour_dir = '{}/{}/{:02d}'.format(image_folder, iter_idx, task_num)
        plt.savefig(behaviour_dir + '_rew_decoder.pdf')
        plt.close()
    else:
        plt.show()


def plot_bb(env, args, episode_all_obs, episode_goals, reward_decoder,
            episode_latent_means, episode_latent_logvars, image_folder, iter_idx, episode_beliefs, task_num,
            episode_prev_obs, episode_actions, episode_rewards):
    """
    Plot behaviour and belief.
    """

    plt.figure(figsize=(1.5 * env._max_episode_steps, 1.5 * args.max_rollouts_per_task))

    num_episodes = len(episode_all_obs)
    num_steps = len(episode_all_obs[0])

    rew_pred_means = [[] for _ in range(num_episodes)]
    rew_pred_vars = [[] for _ in range(num_episodes)]

    # loop through the experiences
    for episode_idx in range(num_episodes):
        for step_idx in range(num_steps):

            curr_obs = episode_all_obs[episode_idx][:step_idx + 1]
            curr_goal = episode_goals[episode_idx]

            if episode_latent_means is not None:
                curr_means = episode_latent_means[episode_idx][:step_idx + 1]
                curr_logvars = episode_latent_logvars[episode_idx][:step_idx + 1]

            # choose correct subplot
            plt.subplot(args.max_rollouts_per_task,
                        math.ceil(env._max_episode_steps) + 1,
                        1 + episode_idx * (1 + math.ceil(env._max_episode_steps)) + step_idx),

            # plot the behaviour
            plot_behaviour(env, curr_obs, curr_goal)

            if reward_decoder is not None:
                # visualise belief in env
                rm, rv = compute_beliefs(env,
                                         args,
                                         reward_decoder,
                                         curr_means[-1],
                                         curr_logvars[-1],
                                         curr_goal)
                rew_pred_means[episode_idx].append(rm)
                rew_pred_vars[episode_idx].append(rv)
                plot_belief(env, rm)
            elif episode_beliefs is not None:
                curr_beliefs = torch.tensor(episode_beliefs[episode_idx][step_idx])
                plot_belief(env, curr_beliefs)
            else:
                rew_pred_means = rew_pred_vars = None

            if episode_idx == 0:
                plt.title('t = {}'.format(step_idx))

            if step_idx == 0:
                plt.ylabel('Episode {}'.format(episode_idx + 1))

    if reward_decoder is not None:
        rew_pred_means = [torch.stack(r) for r in rew_pred_means]
        rew_pred_vars = [torch.stack(r) for r in rew_pred_vars]

    # save figure that shows policy behaviour
    plt.tight_layout()
    if image_folder is not None:
        behaviour_dir = '{}/{}/{:02d}'.format(image_folder, iter_idx, task_num)
        plt.savefig(behaviour_dir + '_behaviour.pdf')
        plt.close()
        if reward_decoder is not None:
            for i in range(num_episodes):
                np.savez(behaviour_dir + '_' + str(i) + '_data.npz',
                         episode_latent_means=episode_latent_means[i].detach().cpu().numpy(),
                         episode_latent_logvars=episode_latent_logvars[i].detach().cpu().numpy(),
                         episode_prev_obs=episode_prev_obs[i].detach().cpu().numpy(),
                         episode_actions=episode_actions[i].detach().cpu().numpy(),
                         episode_rewards=episode_rewards[i].detach().cpu().numpy(),
                         episode_rew_pred_means = rew_pred_means[i].detach().cpu().numpy(),
                         episode_rew_pred_vars = rew_pred_vars[i].detach().cpu().numpy(),
                         )
        else:
            for i in range(num_episodes):
                np.savez(behaviour_dir + '_' + str(i) + '_data.npz',
                         episode_prev_obs=episode_prev_obs[i].detach().cpu().numpy(),
                         episode_actions=episode_actions[i].detach().cpu().numpy(),
                         episode_rewards=episode_rewards[i].detach().cpu().numpy(),
                         )

    else:
        plt.show()

    return rew_pred_means, rew_pred_vars


def plot_behaviour(env, observations, goal):
    num_cells = int(env.observation_space.high[0] + 1)

    # draw grid
    for i in range(num_cells):
        for j in range(num_cells):
            pos_i = i
            pos_j = j
            rec = Rectangle((pos_i, pos_j), 1, 1, facecolor='none', alpha=0.5,
                            edgecolor='k')
            plt.gca().add_patch(rec)

    # shift obs and goal by half a stepsize
    if isinstance(observations, tuple) or isinstance(observations, list):
        observations = torch.cat(observations)
    observations = observations.cpu().numpy() + 0.5
    goal = np.array(goal) + 0.5

    # visualise behaviour, current position, goal
    plt.plot(observations[:, 0], observations[:, 1], 'b-')
    plt.plot(observations[-1, 0], observations[-1, 1], 'b.')
    plt.plot(goal[0], goal[1], 'kx')

    # make it look nice
    plt.xticks([])
    plt.yticks([])
    plt.xlim([0, num_cells])
    plt.ylim([0, num_cells])


def compute_beliefs(env, args, reward_decoder, latent_mean, latent_logvar, goal):
    num_cells = int(env.observation_space.high[0]) + 1
    unwrapped_env = env.venv.unwrapped.envs[0]

    if not args.disable_stochasticity_in_latent:
        # take several samples fromt he latent distribution
        samples = utl.sample_gaussian(latent_mean.view(-1), latent_logvar.view(-1), 100)
    else:
        samples = torch.cat((latent_mean.view(-1), latent_logvar.view(-1))).unsqueeze(0)

    # compute reward predictions for those
    if reward_decoder.multi_head:
        if args.auto_multi_head_num!=0:
            tsm = []
            tsv = []
            for st in range(num_cells ** 2):
                task_id = torch.FloatTensor(unwrapped_env.id_to_task_(torch.tensor([st]))).to(device)
                # curr_state = unwrapped_env.goal_to_onehot_id(task_id).expand((samples.shape[0], 2))
                curr_state = task_id.expand((samples.shape[0], 3))
                if unwrapped_env.oracle:
                    if isinstance(goal, np.ndarray):
                        goal = torch.from_numpy(goal)
                    curr_state = torch.cat((curr_state, goal.repeat(curr_state.shape[0], 1).float()), dim=1)
                tsm.append(torch.mean(reward_decoder(samples, curr_state)))
                tsv.append(torch.var(reward_decoder(samples, curr_state)))
            rew_pred_means = torch.stack(tsm).reshape((1, -1))[0]
            rew_pred_vars = torch.stack(tsv).reshape((1, -1))[0]
        else:
            rew_pred_means = torch.mean(reward_decoder(samples, None), dim=0)  # .reshape((1, -1))
            rew_pred_vars = torch.var(reward_decoder(samples, None), dim=0)  # .reshape((1, -1))]

    else:
        tsm = []
        tsv = []
        for st in range(num_cells ** 2):
            task_id =torch.FloatTensor(unwrapped_env.id_to_task_(torch.tensor([st]))).to(device)
            #curr_state = unwrapped_env.goal_to_onehot_id(task_id).expand((samples.shape[0], 2))
            curr_state = task_id.expand((samples.shape[0], 3))
            if unwrapped_env.oracle:
                if isinstance(goal, np.ndarray):
                    goal = torch.from_numpy(goal)
                curr_state = torch.cat((curr_state, goal.repeat(curr_state.shape[0], 1).float()), dim=1)
            tsm.append(torch.mean(reward_decoder(samples, curr_state)))
            tsv.append(torch.var(reward_decoder(samples, curr_state)))
        rew_pred_means = torch.stack(tsm).reshape((1, -1))[0]
        rew_pred_vars = torch.stack(tsv).reshape((1, -1))[0]
    # rew_pred_means = rew_pred_means[-1][0]
    return rew_pred_means, rew_pred_vars


def plot_belief(env, beliefs):
    """
    Plot the belief by taking 100 samples from the latent space and plotting the average predicted reward per cell.
    """

    num_cells = int(env.observation_space.high[0] + 1)
    unwrapped_env = env.venv.unwrapped.envs[0]

    if not hasattr(unwrapped_env, 'task_to_id'):
        return

    # draw probabilities for each grid cell
    alphas = []
    for i in range(num_cells):
        for j in range(num_cells):
            pos_i = i
            pos_j = j
            idx = unwrapped_env.task_to_id(torch.tensor([[pos_i, pos_j]]))
            alpha = beliefs[idx]
            alpha =torch.clamp(alpha, min=0.0, max=1.0)
            alphas.append(alpha.item())
    alphas = np.array(alphas)
    # alphas = (np.array(alphas)-min(alphas)) / (max(alphas) - min(alphas))
    count = 0
    for i in range(num_cells):
        for j in range(num_cells):
            pos_i = i
            pos_j = j
            rec = Rectangle((pos_i, pos_j), 1, 1, facecolor='r', alpha=alphas[count],
                            edgecolor='k')
            plt.gca().add_patch(rec)
            count += 1
