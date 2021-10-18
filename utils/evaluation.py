import matplotlib.pyplot as plt
import numpy as np
import torch

from environments.parallel_envs import make_vec_envs
from utils import helpers as utl

import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def evaluate(args,
             policy,
             obs_rms,
             ret_rms,
             iter_idx,
             encoder=None,
             num_episodes=None
             ):
    env_name = args.env_name
    if hasattr(args, 'test_env_name'):
        env_name = args.test_env_name
    if num_episodes is None:
        num_episodes = args.max_rollouts_per_task
    num_processes = args.num_processes

    # --- set up the things we want to log ---

    # for each process, we log the returns during the first, second, ... episode
    # (such that we have a minium of [num_episodes]; the last column is for
    #  any overflow and will be discarded at the end, because we need to wait until
    #  all processes have at least [num_episodes] many episodes)

    returns_per_episode = torch.zeros((num_processes, num_episodes + 1)).to(device)

    # --- initialise environments and latents ---

    envs = make_vec_envs(env_name, seed=args.seed * 42 + iter_idx, num_processes=num_processes,
                         gamma=args.policy_gamma, log_dir=args.agent_log_dir,
                         device=device, allow_early_resets=False,
                         rank_offset=num_processes + 1,  # to use diff tmp folders than main processes
                         episodes_per_task=num_episodes,
                         obs_rms=obs_rms, ret_rms=ret_rms)
    num_steps = envs._max_episode_steps

    # reset environments
    (obs_raw, obs_normalised) = envs.reset()
    obs_raw = obs_raw.to(device)
    obs_normalised = obs_normalised.to(device)

    # this counts how often an agent has done the same task already
    task_count = torch.zeros(num_processes).long().to(device)

    if encoder is not None:
        # reset latent state to prior
        latent_sample, latent_mean, latent_logvar, hidden_state = encoder.prior(num_processes)
    else:
        latent_sample = latent_mean = latent_logvar = hidden_state = None

    for episode_idx in range(num_episodes): # num episodes = 4

        for step_idx in range(num_steps): #num_steps = 15

            with torch.no_grad():
                _, action, _ = utl.select_action(args=args,
                                                 policy=policy,
                                                 obs=obs_normalised if args.norm_obs_for_policy else obs_raw,
                                                 latent_sample=latent_sample,
                                                 latent_mean=latent_mean,
                                                 latent_logvar=latent_logvar,
                                                 deterministic=True)

            # observe reward and next obs
            (obs_raw, obs_normalised), (rew_raw, rew_normalised), done, infos = utl.env_step(envs, action)
            done_mdp = [info['done_mdp'] for info in infos]

            if encoder is not None:
                # update the hidden state
                latent_sample, latent_mean, latent_logvar, hidden_state = utl.update_encoding(encoder=encoder,
                                                                                              next_obs=obs_raw,
                                                                                              action=action,
                                                                                              reward=rew_raw,
                                                                                              done=None,
                                                                                              hidden_state=hidden_state)

            # add rewards
            returns_per_episode[range(num_processes), task_count] += rew_raw.view(-1)

            for i in np.argwhere(done_mdp).flatten():
                # count task up, but cap at num_episodes + 1
                task_count[i] = min(task_count[i] + 1, num_episodes)  # zero-indexed, so no +1
            for i in np.argwhere(done).flatten():
                [obs_raw[i], obs_normalised[i]] = envs.reset(index=i)

    envs.close()

    return returns_per_episode[:, :num_episodes]


def visualise_behaviour(args,
                        policy,
                        image_folder,
                        iter_idx,
                        obs_rms, ret_rms,
                        encoder=None,
                        reward_decoder=None,
                        state_decoder=None,
                        task_decoder=None,
                        compute_rew_reconstruction_loss=None,
                        compute_task_reconstruction_loss=None,
                        compute_state_reconstruction_loss=None,
                        compute_kl_loss=None,
                        task_num = None,
                        trial_num = None,
                        encoder_vae = None,
                        args_pol = None,
                        ):
    # initialise environment
    env = make_vec_envs(env_name=args.env_name,
                        seed=args.seed * 42 + iter_idx,
                        num_processes=1,
                        gamma=args.policy_gamma,
                        log_dir=args.agent_log_dir,
                        device=device,
                        allow_early_resets=False,
                        episodes_per_task=args.max_rollouts_per_task,
                        obs_rms=obs_rms, ret_rms=ret_rms,
                        rank_offset=args.num_processes + 42,  # not sure if the temp folders would otherwise clash
                        )


    episode_task = torch.from_numpy(np.array(env.get_task())).to(device).float()

    # get a sample rollout
    unwrapped_env = env.venv.unwrapped.envs[0]
    if hasattr(env.venv.unwrapped.envs[0], 'unwrapped'):
        unwrapped_env = unwrapped_env.unwrapped
    if hasattr(unwrapped_env, 'visualise_behaviour'):
        # if possible, get it from the env directly
        # (this might visualise other things in addition)
        traj = unwrapped_env.visualise_behaviour(env=env,
                                                 args=args,
                                                 policy=policy,
                                                 iter_idx=iter_idx,
                                                 encoder=encoder,
                                                 reward_decoder=reward_decoder,
                                                 state_decoder=state_decoder,
                                                 task_decoder=task_decoder,
                                                 image_folder=image_folder,
                                                 task_num = task_num,
                                                 trial_num = trial_num,
                                                 encoder_vae = encoder_vae,
                                                 args_pol = args_pol,
                                                 )
    else:
        if not args.decode_reward:
            traj = get_test_rollout(args, env, policy, encoder, reward_decoder, image_folder, iter_idx, task_num)
        elif args_pol is None: #varibad case
            traj = get_test_rollout(args, env, policy, encoder, reward_decoder, image_folder, iter_idx, task_num, encoder)
        else: #ldm case
            traj = get_test_rollout(args_pol, env, policy, encoder, reward_decoder, image_folder, iter_idx, task_num, encoder_vae)

    latent_means, latent_logvars, episode_prev_obs, episode_next_obs, episode_actions, episode_rewards, episode_returns = traj

    if latent_means is not None:
        plot_latents(latent_means, latent_logvars,
                     image_folder=image_folder,
                     iter_idx=iter_idx,
                     task_num = task_num,
                     trial_num = trial_num,
                     )

        if not (args.disable_stochasticity_in_latent and args.disable_decoder):
            plot_vae_loss(args,
                          latent_means,
                          latent_logvars,
                          episode_prev_obs,
                          episode_next_obs,
                          episode_actions,
                          episode_rewards,
                          episode_task,
                          image_folder=image_folder,
                          iter_idx=iter_idx,
                          reward_decoder=reward_decoder,
                          state_decoder=state_decoder,
                          task_decoder=task_decoder,
                          compute_task_reconstruction_loss=compute_task_reconstruction_loss,
                          compute_rew_reconstruction_loss=compute_rew_reconstruction_loss,
                          compute_state_reconstruction_loss=compute_state_reconstruction_loss,
                          compute_kl_loss=compute_kl_loss,
                          task_num = task_num,
                          trial_num  = trial_num,
                          )

    env.close()

    #if torch.is_tensor(episode_returns):
    #    return torch.stack(episode_returns).sum(dim=0).cpu().numpy()[0][0]
    #else:
    #    return(sum(episode_returns))
    return torch.stack(episode_returns).sum(dim=0).cpu().numpy()[0][0]


def get_test_rollout(args, env, policy, encoder=None, reward_decoder=None, image_folder=None, iter_idx=None, task_num=None, encoder_vae=None):
    num_episodes = args.max_rollouts_per_task

    # --- initialise things we want to keep track of ---

    episode_prev_obs = [[] for _ in range(num_episodes)]
    episode_next_obs = [[] for _ in range(num_episodes)]
    episode_actions = [[] for _ in range(num_episodes)]
    episode_rewards = [[] for _ in range(num_episodes)]
    if args.env_name == 'HumanoidDir-v0':
        episode_poss = [[] for _ in range(num_episodes)]

    episode_returns = []
    episode_lengths = []

    if encoder is not None:
        episode_latent_samples = [[] for _ in range(num_episodes)]
        episode_latent_means = [[] for _ in range(num_episodes)]
        episode_latent_logvars = [[] for _ in range(num_episodes)]
    else:
        curr_latent_sample = curr_latent_mean = curr_latent_logvar = None
        episode_latent_means = episode_latent_logvars = None

    if encoder_vae is not None:
        # keep track of latent spaces
        episode_latent_samples_vae = [[] for _ in range(num_episodes)]
        episode_latent_means_vae = [[] for _ in range(num_episodes)]
        episode_latent_logvars_vae = [[] for _ in range(num_episodes)]
    else:
        episode_latent_samples_vae = episode_latent_means_vae = episode_latent_logvars_vae = None
    # --- roll out policy ---

    # (re)set environment
    [obs_raw, obs_normalised] = env.reset()
    obs_raw = obs_raw.reshape((1, -1)).to(device)
    obs_normalised = obs_normalised.reshape((1, -1)).to(device)

    for episode_idx in range(num_episodes):

        curr_rollout_rew = []

        if encoder is not None:
            if episode_idx == 0 and encoder:
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

        for step_idx in range(1, env._max_episode_steps + 1):

            episode_prev_obs[episode_idx].append(obs_raw.clone())

            _, action, _ = utl.select_action(args=args,
                                             policy=policy,
                                             obs=obs_normalised if args.norm_obs_for_policy else obs_raw,
                                             deterministic=True,
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

            episode_next_obs[episode_idx].append(obs_raw.clone())
            episode_rewards[episode_idx].append(rew_raw.clone())
            episode_actions[episode_idx].append(action.clone())
            episode_poss[episode_idx].append(infos[0]['pos'])
            curr_rollout_rew.append(rew_raw.clone())

            if infos[0]['done_mdp']:
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
    episode_rewards = [torch.cat(r) for r in episode_rewards]

    behaviour_dir = '{}/{}/{:02d}'.format(image_folder, iter_idx, task_num)
    if 1:
        for i in range(num_episodes):
            if args.env_name == 'HumanoidDir-v0':
                np.savez(behaviour_dir + '_' + str(i) + '_data.npz',
                         episode_poss=episode_poss[i],
                         episode_actions=episode_actions[i].detach().cpu().numpy(),
                         episode_rewards=episode_rewards[i].detach().cpu().numpy(),
                         )
            else:
                np.savez(behaviour_dir + '_' + str(i) + '_data.npz',
                         episode_prev_obs=episode_prev_obs[i][:2].detach().cpu().numpy(),
                         episode_actions=episode_actions[i].detach().cpu().numpy(),
                         episode_rewards=episode_rewards[i].detach().cpu().numpy(),
                         )

    if encoder_vae is not None:
        return episode_latent_means_vae, episode_latent_logvars_vae, \
               episode_prev_obs, episode_next_obs, episode_actions, episode_rewards, \
               episode_returns
    else:
        return episode_latent_means, episode_latent_logvars, \
               episode_prev_obs, episode_next_obs, episode_actions, episode_rewards, \
               episode_returns


def plot_latents(latent_means,
                 latent_logvars,
                 image_folder,
                 iter_idx,
                 task_num,
                 trial_num,
                 ):
    """
    Plot mean/variance over time
    """

    num_rollouts = len(latent_means)
    num_episode_steps = len(latent_means[0])

    latent_means = torch.cat(latent_means).cpu().detach().numpy()
    latent_logvars = torch.cat(latent_logvars).cpu().detach().numpy()

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(latent_means.shape[0]), latent_means, '-', alpha=0.5)
    plt.plot(range(latent_means.shape[0]), latent_means.mean(axis=1), 'k-')
    for tj in np.cumsum([0, *[num_episode_steps for _ in range(num_rollouts)]]):
        span = latent_means.max() - latent_means.min()
        plt.plot([tj + 0.5, tj + 0.5],
                 [latent_means.min() - span * 0.05, latent_means.max() + span * 0.05],
                 'k--', alpha=0.5)
    plt.xlabel('env steps', fontsize=15)
    plt.ylabel('latent mean', fontsize=15)

    plt.subplot(1, 2, 2)
    latent_var = np.exp(latent_logvars)
    plt.plot(range(latent_logvars.shape[0]), latent_var, '-', alpha=0.5)
    plt.plot(range(latent_logvars.shape[0]), latent_var.mean(axis=1), 'k-')
    for tj in np.cumsum([0, *[num_episode_steps for _ in range(num_rollouts)]]):
        span = latent_var.max() - latent_var.min()
        plt.plot([tj + 0.5, tj + 0.5],
                 [latent_var.min() - span * 0.05, latent_var.max() + span * 0.05],
                 'k--', alpha=0.5)
    plt.xlabel('env steps', fontsize=15)
    plt.ylabel('latent variance', fontsize=15)

    plt.tight_layout()
    if image_folder is not None:
        #if trial_num is None:
        #    plt.savefig('{}/{}/{:02d}_latents.pdf'.format(image_folder, iter_idx, task_num))
        #else:
        #    plt.savefig('{}/{}/{}/{:02d}_latents.pdf'.format(image_folder, iter_idx, trial_num, task_num))
        plt.close()
    else:
        plt.show()


def plot_vae_loss(args,
                  latent_means,
                  latent_logvars,
                  prev_obs,
                  next_obs,
                  actions,
                  rewards,
                  task,
                  image_folder,
                  iter_idx,
                  reward_decoder,
                  state_decoder,
                  task_decoder,
                  compute_task_reconstruction_loss,
                  compute_rew_reconstruction_loss,
                  compute_state_reconstruction_loss,
                  compute_kl_loss,
                  task_num,
                  trial_num,
                  ):
    num_rollouts = len(latent_means)
    num_episode_steps = len(latent_means[0])
    if not args.disable_stochasticity_in_latent:
        num_samples = 10  # how many samples to use to get an average/std ELBO loss
    else:
        num_samples = 1

    latent_means = torch.cat(latent_means)
    latent_logvars = torch.cat(latent_logvars)

    prev_obs = torch.cat(prev_obs).to(device)
    next_obs = torch.cat(next_obs).to(device)
    actions = torch.cat(actions).to(device)
    rewards = torch.cat(rewards).to(device)

    # - we will try to make predictions for each tuple in trajectory, hence we need to expand the targets
    prev_obs = prev_obs.unsqueeze(0).expand(num_samples, *prev_obs.shape).to(device)
    next_obs = next_obs.unsqueeze(0).expand(num_samples, *next_obs.shape).to(device)
    actions = actions.unsqueeze(0).expand(num_samples, *actions.shape).to(device)
    rewards = rewards.unsqueeze(0).expand(num_samples, *rewards.shape).to(device)

    rew_reconstr_mean = []
    rew_reconstr_std = []
    rew_pred_std = []

    state_reconstr_mean = []
    state_reconstr_std = []
    state_pred_std = []

    task_reconstr_mean = []
    task_reconstr_std = []
    task_pred_std = []

    # compute the sum of ELBO_t's by looping through (trajectory length + prior)
    for i in range(len(latent_means)):

        curr_latent_mean = latent_means[i]
        curr_latent_logvar = latent_logvars[i]

        # compute the reconstruction loss
        if not args.disable_stochasticity_in_latent:
            # take several samples from the latent distribution
            latent_samples = utl.sample_gaussian(curr_latent_mean.view(-1), curr_latent_logvar.view(-1), num_samples)
        else:
            latent_samples = torch.cat((curr_latent_mean.view(-1), curr_latent_logvar.view(-1))).unsqueeze(0)

        # expand: each latent sample will be used to make predictions for the entire trajectory
        len_traj = prev_obs.shape[1]

        # compute reconstruction losses
        if task_decoder is not None:
            loss_task, task_pred = compute_task_reconstruction_loss(latent_samples, task, return_predictions=True)

            # average/std across the different samples
            task_reconstr_mean.append(loss_task.mean())
            task_reconstr_std.append(loss_task.std())
            task_pred_std.append(task_pred.std())

        latent_samples = latent_samples.unsqueeze(1).expand(num_samples, len_traj, latent_samples.shape[-1])

        if reward_decoder is not None:
            loss_rew, rew_pred = compute_rew_reconstruction_loss(latent_samples, prev_obs, next_obs,
                                                                 actions, rewards, return_predictions=True)
            # sum along length of trajectory
            loss_rew = loss_rew.sum(dim=1)
            rew_pred = rew_pred.sum(dim=1)

            # average/std across the different samples
            rew_reconstr_mean.append(loss_rew.mean())
            rew_reconstr_std.append(loss_rew.std())
            rew_pred_std.append(rew_pred.std())

        if state_decoder is not None:
            loss_state, state_pred = compute_state_reconstruction_loss(latent_samples, prev_obs, next_obs,
                                                                       actions, rewards, return_predictions=True)
            # sum along length of trajectory
            loss_state = loss_state.sum(dim=1)
            state_pred = state_pred.sum(dim=1)

            # average/std across the different samples
            state_reconstr_mean.append(loss_state.mean())
            state_reconstr_std.append(loss_state.std())
            state_pred_std.append(state_pred.std())

    # kl term
    vae_kl_term = compute_kl_loss(latent_means, latent_logvars, None)

    # --- plot KL term ---

    x = range(len(vae_kl_term))

    plt.plot(x, vae_kl_term.cpu().detach().numpy(), 'b-')
    for tj in np.cumsum([0, *[num_episode_steps for _ in range(num_rollouts)]]):
        span = vae_kl_term.max() - vae_kl_term.min()
        plt.plot([tj + 0.5, tj + 0.5],
                 [vae_kl_term.min() - span * 0.05, vae_kl_term.max() + span * 0.05],
                 'k--', alpha=0.5)
    plt.xlabel('env steps', fontsize=15)
    plt.ylabel('KL term', fontsize=15)
    plt.tight_layout()
    if image_folder is not None:
        #if trial_num is None:
        #    plt.savefig('{}/{}/{:02d}_kl.pdf'.format(image_folder, iter_idx, task_num))
        #else:
        #    plt.savefig('{}/{}/{}/{:02d}_kl.pdf'.format(image_folder, iter_idx, trial_num, task_num))
        plt.close()
    else:
        plt.show()

    # --- plot rew reconstruction ---

    if reward_decoder is not None:

        rew_reconstr_mean = torch.stack(rew_reconstr_mean).detach().cpu().numpy()
        rew_reconstr_std = torch.stack(rew_reconstr_std).detach().cpu().numpy()
        rew_pred_std = torch.stack(rew_pred_std).detach().cpu().numpy()

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        p = plt.plot(x, rew_reconstr_mean, 'b-')
        plt.gca().fill_between(x,
                               rew_reconstr_mean - rew_reconstr_std,
                               rew_reconstr_mean + rew_reconstr_std,
                               facecolor=p[0].get_color(), alpha=0.1)
        for tj in np.cumsum([0, *[num_episode_steps for _ in range(num_rollouts)]]):
            min_y = (rew_reconstr_mean - rew_reconstr_std).min()
            max_y = (rew_reconstr_mean + rew_reconstr_std).max()
            span = max_y - min_y
            plt.plot([tj + 0.5, tj + 0.5],
                     [min_y - span * 0.05, max_y + span * 0.05],
                     'k--', alpha=0.5)
        plt.xlabel('env steps', fontsize=15)
        plt.ylabel('reward reconstruction error', fontsize=15)

        plt.subplot(1, 2, 2)
        plt.plot(x, rew_pred_std, 'b-')
        for tj in np.cumsum([0, *[num_episode_steps for _ in range(num_rollouts)]]):
            span = rew_pred_std.max() - rew_pred_std.min()
            plt.plot([tj + 0.5, tj + 0.5],
                     [rew_pred_std.min() - span * 0.05, rew_pred_std.max() + span * 0.05],
                     'k--', alpha=0.5)
        plt.xlabel('env steps', fontsize=15)
        plt.ylabel('std of rew reconstruction', fontsize=15)
        plt.tight_layout()
        if image_folder is not None:
            #if trial_num is None:
            #    plt.savefig('{}/{}/{:02d}_rew_reconstruction.pdf'.format(image_folder, iter_idx, task_num))
            #else:
            #    plt.savefig('{}/{}/{}/{:02d}_rew_reconstruction.pdf'.format(image_folder, iter_idx, trial_num, task_num))
            plt.close()
        else:
            plt.show()

    # --- plot state reconstruction ---

    if state_decoder is not None:

        plt.figure(figsize=(12, 5))

        state_reconstr_mean = torch.stack(state_reconstr_mean).detach().cpu().numpy()
        state_reconstr_std = torch.stack(state_reconstr_std).detach().cpu().numpy()
        state_pred_std = torch.stack(state_pred_std).detach().cpu().numpy()

        plt.subplot(1, 2, 1)
        p = plt.plot(x, state_reconstr_mean, 'b-')
        plt.gca().fill_between(x,
                               state_reconstr_mean - state_reconstr_std,
                               state_reconstr_mean + state_reconstr_std,
                               facecolor=p[0].get_color(), alpha=0.1)
        for tj in np.cumsum([0, *[num_episode_steps for _ in range(num_rollouts)]]):
            min_y = (state_reconstr_mean - state_reconstr_std).min()
            max_y = (state_reconstr_mean + state_reconstr_std).max()
            span = max_y - min_y
            plt.plot([tj + 0.5, tj + 0.5],
                     [min_y - span * 0.05, max_y + span * 0.05],
                     'k--', alpha=0.5)
        plt.xlabel('env steps', fontsize=15)
        plt.ylabel('state reconstruction error', fontsize=15)

        plt.subplot(1, 2, 2)
        plt.plot(x, state_pred_std, 'b-')
        for tj in np.cumsum([0, *[num_episode_steps for _ in range(num_rollouts)]]):
            span = state_pred_std.max() - state_pred_std.min()
            plt.plot([tj + 0.5, tj + 0.5],
                     [state_pred_std.min() - span * 0.05, state_pred_std.max() + span * 0.05],
                     'k--', alpha=0.5)
        plt.xlabel('env steps', fontsize=15)
        plt.ylabel('std of state reconstruction', fontsize=15)
        plt.tight_layout()
        if image_folder is not None:
            #if trial_num is None:
            #    plt.savefig('{}/{}/{:02d}_state_reconstruction.pdf'.format(image_folder, iter_idx, task_num))
            #else:
            #    plt.savefig('{}/{}/{}/{:02d}_state_reconstruction.pdf'.format(image_folder, iter_idx, trial_num, task_num))
            plt.close()
        else:
            plt.show()

    # --- plot task reconstruction ---

    if task_decoder is not None:

        plt.figure(figsize=(12, 5))

        task_reconstr_mean = torch.stack(task_reconstr_mean).detach().cpu().numpy()
        task_reconstr_std = torch.stack(task_reconstr_std).detach().cpu().numpy()
        task_pred_std = torch.stack(task_pred_std).detach().cpu().numpy()

        plt.subplot(1, 2, 1)
        p = plt.plot(x, task_reconstr_mean, 'b-')
        plt.gca().fill_between(x,
                               task_reconstr_mean - task_reconstr_std,
                               task_reconstr_mean + task_reconstr_std,
                               facecolor=p[0].get_color(), alpha=0.1)
        for tj in np.cumsum([0, *[num_episode_steps for _ in range(num_rollouts)]]):
            min_y = (task_reconstr_mean - task_reconstr_std).min()
            max_y = (task_reconstr_mean + task_reconstr_std).max()
            span = max_y - min_y
            plt.plot([tj + 0.5, tj + 0.5],
                     [min_y - span * 0.05, max_y + span * 0.05],
                     'k--', alpha=0.5)
        plt.xlabel('env steps', fontsize=15)
        plt.ylabel('task reconstruction error', fontsize=15)

        plt.subplot(1, 2, 2)
        plt.plot(x, task_pred_std, 'b-')
        for tj in np.cumsum([0, *[num_episode_steps for _ in range(num_rollouts)]]):
            span = task_pred_std.max() - task_pred_std.min()
            plt.plot([tj + 0.5, tj + 0.5],
                     [task_pred_std.min() - span * 0.05, task_pred_std.max() + span * 0.05],
                     'k--', alpha=0.5)
        plt.xlabel('env steps', fontsize=15)
        plt.ylabel('std of task reconstruction', fontsize=15)
        plt.tight_layout()
        if image_folder is not None:
            #plt.savefig('{}/{}/{:02d}_task_reconstruction.pdf'.format(image_folder, iter_idx, task_num))
            plt.close()
        else:
            plt.show()
