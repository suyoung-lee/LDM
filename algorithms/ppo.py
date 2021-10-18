import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import helpers as utl


class PPO:
    def __init__(self,
                 actor_critic,
                 value_loss_coef,
                 entropy_coef,
                 optimiser_vae=None,
                 lr=None,
                 clip_param=0.2,
                 ppo_epoch=5,
                 num_mini_batch=5,
                 eps=None,
                 use_huber_loss=True,
                 use_clipped_value_loss=True,
                 ):

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.use_clipped_value_loss = use_clipped_value_loss
        self.use_huber_loss = use_huber_loss

        self.optimiser = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)
        self.optimiser_vae = optimiser_vae

    def update(self,
               args,
               policy_storage,
               encoder=None,  # variBAD encoder
               rlloss_through_encoder=False,  # whether or not to backprop RL loss through encoder
               compute_vae_loss=None  # function that can compute the VAE loss
               ):

        # -- get action values --
        advantages = policy_storage.returns[:-1] - policy_storage.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        # if this is true, we will update the VAE at every PPO update
        # otherwise, we update it after we update the policy
        if rlloss_through_encoder:
            # recompute embeddings (to build computation graph)
            utl.recompute_embeddings(policy_storage, encoder, sample=False, update_idx=0)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        loss_epoch = 0
        for e in range(self.ppo_epoch):

            data_generator = policy_storage.feed_forward_generator(advantages, self.num_mini_batch)
            for sample in data_generator:

                obs_batch, actions_batch, latent_sample_batch, latent_mean_batch, latent_logvar_batch, \
                value_preds_batch, return_batch, old_action_log_probs_batch, \
                adv_targ = sample

                if not rlloss_through_encoder:
                    obs_batch = obs_batch.detach()
                    if latent_sample_batch is not None:
                        latent_sample_batch = latent_sample_batch.detach()
                        latent_mean_batch = latent_mean_batch.detach()
                        latent_logvar_batch = latent_logvar_batch.detach()

                obs_aug = utl.get_augmented_obs(args, obs_batch,
                                                latent_sample=latent_sample_batch,
                                                latent_mean=latent_mean_batch,
                                                latent_logvar=latent_logvar_batch, )

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, action_mean, action_logstd = \
                    self.actor_critic.evaluate_actions(obs_aug, actions_batch, return_action_mean=True)

                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_huber_loss and self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                                self.clip_param)
                    value_losses = F.smooth_l1_loss(values, return_batch, reduction='none')
                    value_losses_clipped = F.smooth_l1_loss(value_pred_clipped, return_batch, reduction='none')
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                elif self.use_huber_loss:
                    value_loss = F.smooth_l1_loss(values, return_batch)
                elif self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                                self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                # zero out the gradients
                self.optimiser.zero_grad()
                if rlloss_through_encoder:
                    self.optimiser_vae.zero_grad()

                # compute policy loss and backprop
                loss = value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef

                # compute vae loss and backprop
                if rlloss_through_encoder:
                    loss += args.vae_loss_coeff * compute_vae_loss()

                # compute gradients (will attach to all networks involved in this computation)
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), args.policy_max_grad_norm)
                if (encoder is not None) and rlloss_through_encoder:
                    nn.utils.clip_grad_norm_(encoder.parameters(), args.policy_max_grad_norm)

                # update
                self.optimiser.step()
                if rlloss_through_encoder:
                    self.optimiser_vae.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                loss_epoch += loss.item()

                if rlloss_through_encoder:
                    # recompute embeddings (to build computation graph)
                    utl.recompute_embeddings(policy_storage, encoder, sample=False, update_idx=e + 1)

        if (not rlloss_through_encoder) and (self.optimiser_vae is not None):
            for _ in range(args.num_vae_updates):
                compute_vae_loss(update=True)

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        loss_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, loss_epoch

    def act(self, obs, deterministic=False):
        return self.actor_critic.act(obs, deterministic=deterministic)
