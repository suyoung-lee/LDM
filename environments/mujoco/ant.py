import matplotlib.pyplot as plt
import numpy as np
import torch

from environments.mujoco.mujoco_env import MujocoEnv
from utils import helpers as utl
from matplotlib.patches import Wedge

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AntEnv(MujocoEnv):
    def __init__(self, use_low_gear_ratio=False):
        self.init_serialization(locals())
        #list for testing
        #self.task_list = [-1.0, 1.0]

        if use_low_gear_ratio:
            xml_path = 'low_gear_ratio_ant.xml'
        else:
            xml_path = 'ant.xml'
        super().__init__(
            xml_path,
            frame_skip=5,
            automatically_set_obs_and_action_space=True,
        )

    def step(self, a):
        torso_xyz_before = self.get_body_com("torso")
        self.do_simulation(a, self.frame_skip)
        torso_xyz_after = self.get_body_com("torso")
        torso_velocity = torso_xyz_after - torso_xyz_before
        forward_reward = torso_velocity[0] / self.dt
        ctrl_cost = .5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 0.  # 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()

        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
            torso_velocity=torso_velocity,
        )

    def _get_obs(self):
        # this is gym ant obs, should use rllab?
        # if position is needed, override this in subclasses
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def reset_task(self, task):
        if task is None:
            task = self.sample_tasks(1)[0]
            self.set_task(task)
        else:
            self.set_task(self.get_test_task_list()[task])

    @staticmethod
    def visualise_behaviour(env,
                            args,
                            policy,
                            iter_idx,
                            encoder=None,
                            image_folder=None,
                            return_pos=False,
                            task_num = None,
                            trial_num = None,
                            encoder_vae = None,
                            args_pol = None,
                            **kwargs
                            ):

        num_episodes = args.max_rollouts_per_task
        unwrapped_env = env.venv.unwrapped.envs[0].unwrapped

        # --- initialise things we want to keep track of ---

        episode_prev_obs = [[] for _ in range(num_episodes)]
        episode_next_obs = [[] for _ in range(num_episodes)]
        episode_actions = [[] for _ in range(num_episodes)]
        episode_rewards = [[] for _ in range(num_episodes)]

        episode_returns = []
        episode_lengths = []

        if encoder is not None:
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

        # --- roll out policy ---

        # (re)set environment
        #env.reset_task()

        (obs_raw, obs_normalised) = env.reset()
        env.reset_task(task=task_num)
        obs_raw = obs_raw.float().reshape((1, -1)).to(device)
        obs_normalised = obs_normalised.float().reshape((1, -1)).to(device)
        start_obs_raw = obs_raw.clone()

        # initialise actions and rewards (used as initial input to policy if we have a recurrent policy)
        if hasattr(args, 'hidden_size'):
            hidden_state = torch.zeros((1, args.hidden_size)).to(device)
        else:
            hidden_state = None

        # keep track of what task we're in and the position of the cheetah
        task = env.get_task()
        pos = [[] for _ in range(args.max_rollouts_per_task)]
        start_pos = unwrapped_env.get_body_com("torso")[:2].copy()

        for episode_idx in range(num_episodes):

            curr_rollout_rew = []
            pos[episode_idx].append(start_pos)

            if episode_idx == 0:
                if encoder is not None:
                    # reset to prior
                    curr_latent_sample, curr_latent_mean, curr_latent_logvar, hidden_state = encoder.prior(1)
                    curr_latent_sample = curr_latent_sample[0].to(device)
                    curr_latent_mean = curr_latent_mean[0].to(device)
                    curr_latent_logvar = curr_latent_logvar[0].to(device)
                else:
                    curr_latent_sample = curr_latent_mean = curr_latent_logvar = None

                if encoder_vae is not None:
                    curr_latent_sample_vae, curr_latent_mean_vae, curr_latent_logvar_vae, hidden_state_vae = encoder_vae.prior(1)
                    curr_latent_sample_vae = curr_latent_sample_vae[0].to(device)
                    curr_latent_mean_vae = curr_latent_mean_vae[0].to(device)
                    curr_latent_logvar_vae = curr_latent_logvar_vae[0].to(device)

            if encoder is not None:
                episode_latent_samples[episode_idx].append(curr_latent_sample[0].clone())
                episode_latent_means[episode_idx].append(curr_latent_mean[0].clone())
                episode_latent_logvars[episode_idx].append(curr_latent_logvar[0].clone())

            if encoder_vae is not None:
                episode_latent_samples_vae[episode_idx].append(curr_latent_sample_vae[0].clone())
                episode_latent_means_vae[episode_idx].append(curr_latent_mean_vae[0].clone())
                episode_latent_logvars_vae[episode_idx].append(curr_latent_logvar_vae[0].clone())

            for step_idx in range(1, env._max_episode_steps + 1):

                if step_idx == 1:
                    episode_prev_obs[episode_idx].append(start_obs_raw.clone())
                else:
                    episode_prev_obs[episode_idx].append(obs_raw.clone())
                # act
                if args_pol == None:
                    o_aug = utl.get_augmented_obs(args,
                                                  obs_normalised if args.norm_obs_for_policy else obs_raw,
                                                  curr_latent_sample, curr_latent_mean,
                                                  curr_latent_logvar)
                else: #maybe this is unnecessary
                    o_aug = utl.get_augmented_obs(args_pol,
                                                  obs_normalised if args.norm_obs_for_policy else obs_raw,
                                                  curr_latent_sample, curr_latent_mean,
                                                  curr_latent_logvar)

                _, action, _ = policy.act(o_aug, deterministic=True)
                #print("epi", "step", episode_idx, step_idx)
                (obs_raw, obs_normalised), (rew_raw, rew_normalised), done, info = env.step(action.cpu().detach()) ##varibad wrapper on, so done means all episodes are finished, info done mdp means just current
                #print("done", "done mdp", "rew raw", done, info[0]['done_mdp'], rew_raw)

                obs_raw = obs_raw.float().reshape((1, -1)).to(device)
                obs_normalised = obs_normalised.float().reshape((1, -1)).to(device)

                # keep track of position
                pos[episode_idx].append(unwrapped_env.get_body_com("torso")[:2].copy())

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

                episode_next_obs[episode_idx].append(obs_raw.clone())
                episode_rewards[episode_idx].append(rew_raw.clone())
                episode_actions[episode_idx].append(action.clone())
                curr_rollout_rew.append(rew_raw.clone())

                ##varibad wrapper on, so done means all episodes are finished, info done mdp means just current
                #if info[0]['done_mdp'] and not done:
                if info[0]['done_mdp'] and not done:
                    start_obs_raw = info[0]['start_state']
                    start_obs_raw = torch.from_numpy(start_obs_raw).float().reshape((1, -1)).to(device)
                    start_pos = unwrapped_env.get_body_com("torso")[:2].copy()
                    break
                elif done:
                    break

            episode_returns.append(sum(curr_rollout_rew))
            episode_lengths.append(step_idx)

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

        # plot the movement of the ant
        # print(pos)
        plt.figure(figsize=(4.25, 4 * num_episodes))
        min_dim = -3.5
        max_dim = 3.5
        span = max_dim - min_dim

        for i in range(num_episodes):
            axes = plt.subplot(num_episodes, 1, i + 1)

            if args.env_name == 'AntDir-v0':
                plt.title('task: {} degrees, return: {:04.2f}'.format(np.degrees(task), episode_returns[i].cpu().numpy()[0][0]), fontsize=15)
            if args.env_name == 'AntGoal-v0':
                plt.plot(task[0], task[1], 'rx')
                circle1 = plt.Circle((0.0, 0.0), radius=3.0, facecolor='lightgrey', edgecolor ='black', linestyle='--', zorder=-3)
                circle2 = plt.Circle((0.0, 0.0), radius=2.5, facecolor='white', edgecolor ='black', linestyle='--', zorder=-2)
                circle3 = plt.Circle((0.0, 0.0), radius=1.0, facecolor='lightgrey', edgecolor ='black', linestyle='--', zorder=-1)
                axes.add_artist(circle1)
                axes.add_artist(circle2)
                axes.add_artist(circle3)

                plt.title('return: {:04.2f}'.format(episode_returns[i].cpu().numpy()[0][0], fontsize=15))

            plt.ylabel('y-position (ep {})'.format(i), fontsize=15)

            x = list(map(lambda p: p[0], pos[i]))
            y = list(map(lambda p: p[1], pos[i]))
            plt.plot(x[0], y[0], 'bo')

            plt.scatter(x, y, 1, 'g')

            if i == num_episodes - 1:
                plt.xlabel('x-position', fontsize=15)
                plt.ylabel('y-position (ep {})'.format(i), fontsize=15)
            plt.xlim(min_dim - 0.05 * span, max_dim + 0.05 * span)
            plt.ylim(min_dim - 0.05 * span, max_dim + 0.05 * span)

        plt.tight_layout()
        if image_folder is not None:
            if trial_num is None:
                behaviour_dir = '{}/{}/{:02d}'.format(image_folder, iter_idx, task_num)
            else:
                behaviour_dir = '{}/{}/{}/{:02d}'.format(image_folder, iter_idx, trial_num, task_num)
            plt.savefig(behaviour_dir+'_behaviour.pdf')
            plt.close()

            for i in range(num_episodes):
                if encoder_vae is not None:
                    np.savez(behaviour_dir + '_' + str(i) + '_data.npz',
                             episode_latent_means_vae=episode_latent_means_vae[i].detach().cpu().numpy(),
                             episode_latent_logvars_vae=episode_latent_logvars_vae[i].detach().cpu().numpy(),
                             #episode_prev_obs=episode_prev_obs[i].detach().cpu().numpy(),
                             #episode_next_obs=episode_next_obs[i].detach().cpu().numpy(),
                             #episode_actions=episode_actions[i].detach().cpu().numpy(),
                             episode_rewards=episode_rewards[i].detach().cpu().numpy(),
                             episode_returns=episode_returns[i].detach().cpu().numpy(),
                             episode_position= pos[i]
                             )
                else:
                    np.savez(behaviour_dir + '_' + str(i) + '_data.npz',
                             episode_latent_means=episode_latent_means[i].detach().cpu().numpy(),
                             #episode_latent_logvars=episode_latent_logvars[i].detach().cpu().numpy(),
                             #episode_prev_obs=episode_prev_obs[i].detach().cpu().numpy(),
                             #episode_next_obs=episode_next_obs[i].detach().cpu().numpy(),
                             #episode_actions=episode_actions[i].detach().cpu().numpy(),
                             episode_rewards=episode_rewards[i].detach().cpu().numpy(),
                             episode_returns=episode_returns[i].detach().cpu().numpy(),
                             episode_position=pos[i]
                             )
        else:
            plt.show()

        if not return_pos:
            if encoder_vae is not None:
                return episode_latent_means_vae, episode_latent_logvars_vae, \
                       episode_prev_obs, episode_next_obs, episode_actions, episode_rewards, \
                       episode_returns
            else:
                return episode_latent_means, episode_latent_logvars, \
                       episode_prev_obs, episode_next_obs, episode_actions, episode_rewards, \
                       episode_returns
        else:
            if encoder_vae is not None:
                return episode_latent_means_vae, episode_latent_logvars_vae, \
                       episode_prev_obs, episode_next_obs, episode_actions, episode_rewards, \
                       episode_returns, pos
            else:
                return episode_latent_means, episode_latent_logvars, \
                       episode_prev_obs, episode_next_obs, episode_actions, episode_rewards, \
                       episode_returns, pos
